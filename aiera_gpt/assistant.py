from openai import OpenAI
import time
import json
import openai
import tiktoken


from openai.types.beta.threads import MessageContentText, ThreadMessage
from aiera.shared_services.db import AieraReadDatabase
from aiera_gpt.config import AieraSettings, OpenAISettings, aiera_settings
import logging

logger = logging.getLogger("aiera_gpt.assistant")
logger.addHandler(logging.StreamHandler())
logger.setLevel("DEBUG")

def verify_user(settings: AieraSettings):
    if not settings.api_key:
        logger.debug("User does not have API key defined.")
        return False
    
    else: 
        logger.debug("User has API key defined.")
        return True



# logit bias for trending topics
# annotated text

class Tokenizer():

    def __init__(self, model_name):

        self.encoding = None

        if "gpt-4" in model_name:
            self.encoding = tiktoken.encoding_for_model("gpt-4")

        elif "gpt-3-turbo" in model_name:
            self.encoding = tiktoken.encoding_for_model("gpt-3-turbo")

    def get_token_count(self, messages: list):
        tokens_per_message = 4
        # every message follows <|start|>{role/name}\n{content}<|end|>\n

        token_count = 0
        for each in messages:
            token_count += tokens_per_message
            # processed
            if isinstance(each.content[0], MessageContentText):
                token_count += self.get_text_token_count(each.content[0].text.value)
            else:
                logger.error("Unhandled content type %s", type(each.content[0]))

        return token_count

    def tokenize_text(self, text: str):
        encoding = tiktoken.encoding_for_model("gpt-4")
        return encoding.encode(text)

    def get_text_token_count(self, text: str):
        tokens = self.tokenize_text(text)
        return len(tokens)


class AieraGPTAssistant:

    def __init__(self, settings: OpenAISettings, db_config, aiera_settings):
        self.client = OpenAI(
            organization = settings.org_id,
            api_key = settings.api_token
        )

        openai.organization = settings.org_id
        openai.api_key = settings.api_token

        self.assistant_id = settings.assistant_id
        self.assistant = self.client.beta.assistants.retrieve(self.assistant_id)
        self.thread = self.client.beta.threads.create()
        self.is_verified = verify_user(aiera_settings)
        self.model_name = self.assistant.model
        self.tokenizer = Tokenizer(self.assistant.model)
        self.total_token_count = 0

        # remove once using api
        self.db = AieraReadDatabase(db_config)

    def get_token_count(self, messages: list):
        return self.tokenizer.get_token_count(messages)


    def get_possible_events(self, company: str , quarter= None, year = None):
        # THIS MUST BE IMPLEMENTED IN aiera_api

        quarter_sql = ""
        if quarter is not None:
            quarter_sql = f" AND sac.fiscal_quarter = {quarter}"

        year_sql = ""
        if year is not None:
            year_sql = f" AND sac.fiscal_year = {year}"


        sql = f"""
        SELECT sac.scheduled_audio_call_id, e.equity_id, sac.fiscal_quarter, sac.fiscal_year, sac.transcript_current_version FROM equities e
        JOIN scheduled_audio_calls_nc sac 
        ON e.equity_id = sac.equity_id 
        WHERE e.common_name LIKE '{company}%%'
        AND sac.call_type = 'earnings'
        {quarter_sql}
        {year_sql};
        """

        events = self.db.select_all(sql)
        return events
    
    
    def get_event_transcript(self, company: str, quarter= None, year = None):

        possible_events = self.get_possible_events(company, quarter=quarter, year=year)
        event = possible_events[0]

        sql = """SELECT sace.event_id, COALESCE(sace.transcript_corrected, sace.transcript) AS transcript,
                        COALESCE(p.parent_id, p.person_id) AS person_id, p.name as person_name,
                        sace.transcript_section
                    FROM scheduled_audio_call_events_nc sace
                        LEFT JOIN transcript_speakers ts ON ts.speaker_id = sace.transcript_speaker_id
                        LEFT JOIN persons p ON ts.person_id = p.person_id
                    WHERE sace.scheduled_audio_call_id = %s
                        AND (sace.status IS NULL OR sace.status = 'active')
                        AND sace.event_type IN ('transcript', 'official_transcript')
                        AND sace.transcript_version = %s
                    ORDER BY start_ms ASC"""
        
        transcript_segments = self.db.select_all(sql, event.scheduled_audio_call_id, event.transcript_current_version)

        text = ""
        for segment in transcript_segments:
            chunks = segment.transcript.split("\n")
            text += "\n".join([chunk.strip(" \n") for chunk in chunks])

        return text


    def introduce_self_unverified(self):
        message_content = """Hello, my name is Aiera. Only
            Aiera API subscribers have access to Aiera's transcript database. Visit us at aiera.com to \
            learn more. In the meantime, would you like to explore Apple's latest Earnings Call?'. \
        """
        #message = self.client.beta.threads.messages.create(
        #    thread_id = self.thread.id,
        #    role = "assistant",
        #    content = message_content,
        #)
        message = [{"role": "assistant", "content": message_content}]
        return message

    def introduce_self_verified(self): 
        message_content = """Hello, my name is Aiera. What can I help you with today?\n\n1. Analyze an earnings call"""
        #message = self.client.beta.threads.messages.create(
        #    thread_id = self.thread.id,
        #    role = "assistant",
        #    content = message_content,
        #)
        message = [{"role": "assistant", "content": message_content}]

        return message


    def introduce_self(self):
        if self.is_verified:
            return self.introduce_self_verified()
        
        else:
            return self.introduce_self_unverified()

    def get_default_event_transcript(self):
        message_content = """Only
            Aiera API subscribers have access to Aiera's transcript database. Visit us at aiera.com to \
            learn more. In the meantime, would you like to explore Apple's latest Earnings Call?'. \
        """

        self.client.beta.threads.messages.create(
            thread_id = self.thread.id,
            role = "system",
            content = message_content,
        )
        return [{"role": "assistant", "content": message_content}]

    def submit_message(self, message_content: str):
        logger.debug("Adding message from user to the message thread.")
        self.client.beta.threads.messages.create(
            thread_id = self.thread.id,
            role = "user",
            content = message_content,
        )
        

    def perform_swot(self):
        run = self.client.beta.threads.runs.create(
            thread_id = self.thread.id,
            assistant_id = self.assistant.id,
            instructions = "Load the transcript. Perform a SWOT analysis on the transcript that isolates the strengths, \
            weaknesses, opportunities, threats, and forward outlook. Mention any significant metrics reported \
            during the call. Cite the source for each item in the original text."
        )

        self.process_messages()


    def process_messages(self):
        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        logger.debug("Current messages in thread:\n%s", json.dumps([{"role": mess.role, "content": mess.content[0].text.value} for mess in messages]))


        logger.debug("Creating a remote run using the thread and assistant.")
        run = self.client.beta.threads.runs.create(
            thread_id = self.thread.id,
            assistant_id = self.assistant.id,
        )

        run = self._wait_for_run_event(run)

        if run.status == 'requires_action':
            logger.debug("Action required...")
            tools_to_call = run.required_action.submit_tool_outputs.tool_calls

            tool_output_array = []
            for each_tool in tools_to_call:
                tool_call_id = each_tool.id
                function_name = each_tool.function.name
                function_arg = each_tool.function.arguments
                if function_arg is not None:
                    function_arg = json.loads(function_arg)

                logger.debug("Will attempt to run %s with args %s", function_name, json.dumps(function_arg))

                if function_name == "get_event_transcript":
                    output = self.get_event_transcript(
                        function_arg["company"],
                        quarter = function_arg["quarter"],
                        year = function_arg["year"]
                    )

                tool_output_array.append({"tool_call_id": tool_call_id, "output": output})

            run = self.client.beta.threads.runs.submit_tool_outputs(
                thread_id = self.thread.id,
                run_id = run.id,
                tool_outputs = tool_output_array
            )

            logger.debug("Monitoring status of assistant.")
            run = self._wait_for_run_event(run)

            messages = self.client.beta.threads.messages.list(
                thread_id = self.thread.id
            )

        if run.status == "completed":
            logger.debug("Completed run.")
            messages = self.client.beta.threads.messages.list(
                thread_id = self.thread.id
            )

            # update total token count
            self.total_token_count += self.get_token_count(messages)

            formatted_messages = self._format_openai_messages(messages)
            logger.debug("Current messages:\n%s", json.dumps(formatted_messages))

            return formatted_messages
        
        elif run.status == "failed":
            return []
        
        else:
            logger.error("Something went wrong. Run status : %s", run.status)
        

    def _wait_for_run_event(self, run):
        logger.debug("Monitoring status of assistant run.")
        i = 0
        while run.status not in ["completed", "failed", "requires_action"]:
            logger.debug("Assistant status is %s. Waiting...", run.status)
            if i > 0:
                time.sleep(2)

            run = self.client.beta.threads.runs.retrieve(
                thread_id = self.thread.id,
                run_id = run.id
            )
            i += 1

        return run


    def close_chat(self):
        self.client.beta.threads.delete(self.thread.id)
        self.total_token_count = 0

    def _format_openai_messages(self, messages):

        new_messages = []
        for message in messages:
            if isinstance(message, ThreadMessage):
                new_message = {"role": message.role, "content": message.content[0].text.value}
                new_messages.append(new_message)

            elif isinstance(message, str):
                new_messages.append(message)

        return new_messages
            