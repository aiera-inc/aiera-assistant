from openai import OpenAI
import time
import os
from typing import List
import json
import openai
import tiktoken


from openai.types.beta.threads import MessageContentText, ThreadMessage
from aiera_gpt.config import AieraSettings, OpenAISettings
import logging
import requests

logger = logging.getLogger("aiera_gpt.assistant")

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

    def __init__(self, openai_settings: OpenAISettings, aiera_settings):
        self.client = OpenAI(
            organization = openai_settings.org_id,
            api_key = openai_settings.api_token
        )

        #openai.organization = openai_settings.org_id
        #openai.api_key = openai_settings.api_token

        self.assistant_id = openai_settings.assistant_id
        self.assistant = self.client.beta.assistants.retrieve(self.assistant_id)
        self.thread = self.client.beta.threads.create()
        self.is_verified = verify_user(aiera_settings)
        self.model_name = self.assistant.model
        self.tokenizer = Tokenizer(self.assistant.model)
        self.total_token_count = 0
        self.persist_files = openai_settings.persist_files
        self.aiera_settings = aiera_settings

        self.file_ids = []

        # remove once using api
        #self.db = AieraReadDatabase(db_config)

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
        SELECT sac.scheduled_audio_call_id, e.equity_id, sac.fiscal_quarter, sac.fiscal_year, sac.transcript_current_version, e.search_name FROM equities e
        JOIN scheduled_audio_calls_nc sac 
        ON e.equity_id = sac.equity_id 
        WHERE e.search_name LIKE '{company}%%'
        AND sac.call_type = 'earnings'
        {quarter_sql}
        {year_sql};
        """

        events = self.db.select_all(sql)
        return events
    
    def find_events(self):
        matches = requests.get(f"{self.aiera_settings.base_url}/matches?size=10", 
                    headers={"X-API-Key": self.aiera_settings.api_key})

        event_type = "earnings"
        isin = isin
        with_transcripts = True


        # event will return     "fiscal_year": 2022,
  #  "fiscal_quarter": 3

    def get_event_transcript(self, equity_id: int, quarter: int = None, year: int = None):
        params_string = f"size=10&event_type=earnings&with_transcripts=1&equity_id={equity_id}"
        if quarter is not None:
            params_string += f"&fiscal_quarter={quarter}"
        
        if year is not None:
            params_string += f"&fiscal_year={year}"


        matches = requests.get(f"{self.aiera_settings.base_url}?{params_string}", 
                               headers={"X-API-Key": self.aiera_settings.api_key})

        print(matches)

        breakpoint()


        #transcript = f"{event.search_name} Q{event.fiscal_quarter} Q{event.fiscal_year}\n"
        #for segment in transcript_segments:
        #    chunks = segment.transcript.split("\n")
        #    if segment.person_name:
        #        transcript += f"{segment.person_name}: "
        #    transcript += "\n".join([chunk.strip(" \n") for chunk in chunks])
        #    transcript += "\n"

        #self.upload_transcript_file(event.search_name, event.fiscal_quarter, event.fiscal_year, transcript)


        return None


    
    def upload_transcript_file(self, company, quarter, year, transcript):
        # create temporary local file
        filename = f"{company.replace(' ', '')}_Q{quarter}_{year}.txt"
        with open(filename, "w") as f:
            f.write(transcript)

        #upload a file with an assistants purpose
        try:
            file = self.client.files.create(
                file = open(filename, "r"),
                purpose = "assistants"
            )
            self.file_ids.append(file.id)
        except Exception as e:
            logger.error(str(e))

        # remove local file 
        os.remove(filename)


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
           # file_ids = []
        )
        return [{"role": "assistant", "content": message_content}]

    def submit_message(self, message_content: str):
        logger.debug("Adding message from user to the message thread. Will reference the file_ids: %s", ", ".join(self.file_ids))
        self.client.beta.threads.messages.create(
            thread_id = self.thread.id,
            role = "user",
            content = message_content,
            file_ids = self.file_ids
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


    def process_messages(self) -> List[dict]:
        """Main workflow for processing messages with openai endpoints.
        
        """
        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        logger.debug("Current messages in thread:\n%s", json.dumps([{"role": mess.role, "content": mess.content[0].text.value} for mess in messages]))


        logger.debug("Creating a remote run using the thread and assistant.")
        run = self.client.beta.threads.runs.create(
            thread_id = self.thread.id,
            assistant_id = self.assistant.id
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
                        function_arg["equity_id"],
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
    
    def __del__(self):
        if self.file_ids:
            if self.persist_files:
                logger.info("Files are being persisted.")

            if not self.persist_files:
                logger.info("Configured to remove files uploaded to openai. Deleting...")
                for file_id in self.file_ids:
                    logger.debug("Removing file %s...", file_id)
                    res = self.client.files.delete(
                        file_id = file_id
                    )
