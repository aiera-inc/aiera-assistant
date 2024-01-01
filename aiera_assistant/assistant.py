from openai import OpenAI
import time
import re
from typing import List
import json
import tiktoken
import sqlite3

from openai.types.beta.threads import ThreadMessage
from openai.types.beta.threads.message_content_text import TextAnnotationFileCitation
from aiera_gpt.config import AieraSettings, OpenAISettings
from aiera_assistant.__init__ import ROOT_DIR
import logging
import requests

logger = logging.getLogger("aiera_gpt.assistant")

def verify_user(settings: AieraSettings):
    """
    Verify user settings.

    Args:
        settings (AieraSettings): An object containing user settings.

    Returns:
        bool: True if the user has an API key defined, False otherwise.
    """    
    if not settings.api_key:
        logger.debug("User does not have API key defined.")
        return False
    
    else: 
        logger.debug("User has API key defined.")
        return True



# logit bias for trending topics
# annotated text

class Tokenizer():

    """
    A class for tokenizing text.

    Attributes:
        model_name (str): The name of the model to be used for tokenization.
        encoding (tiktoken.Encoding): The encoding object for the specified model.
    """    
    def __init__(self, model_name):

        """
        Initialize the object with a model name and set the encoding attribute.

        Args:
            model_name (str): The name of the GPT model.

        Raises:
            None.

        Returns:
            None.
        """        
        self.encoding = None

        if "gpt-4" in model_name:
            self.encoding = tiktoken.encoding_for_model("gpt-4")

        elif "gpt-3-turbo" in model_name:
            self.encoding = tiktoken.encoding_for_model("gpt-3-turbo")

    def get_token_count(self, messages: List[dict]):
        """
        Get the total number of tokens in a list of messages.

        Args:
            messages (List[dict]): A list of dictionaries representing messages.

        Returns:
            int: The total number of tokens in the messages.

        Raises:
            None
        """        
        tokens_per_message = 4
        # every message follows <|start|>{role/name}\n{content}<|end|>\n

        token_count = 0
        for each in messages:
            token_count += tokens_per_message
            # processed
            if isinstance(each, dict):
                token_count += self.get_text_token_count(each["content"])
            else:
                logger.error("Unhandled message type %s", type(each))

        return token_count

    def tokenize_text(self, text: str):
        """
        Tokenize the input text using the GPT-4 model.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            List[int]: The tokenized representation of the input text.
        """        
        encoding = tiktoken.encoding_for_model("gpt-4")
        return encoding.encode(text)

    def get_text_token_count(self, text: str):
        """
        Get the number of tokens in a text.

        Args:
            self: The object calling the method.
            text (str): The input text.

        Returns:
            int: The number of tokens in the text.
        """        
        tokens = self.tokenize_text(text)
        return len(tokens)


class AieraAssistant:

    """
    A class representing an Aiera Assistant.

    Attributes:
        openai_settings (OpenAISettings): The settings for the OpenAI API.
        aiera_settings: The settings for the Aiera API.
        db_settings: The settings for the database.
        client: An instance of the OpenAI class.
        assistant_id (str): The ID of the assistant.
        assistant: The assistant object.
        thread: The thread object.
        is_verified (bool): Indicates if the user is verified.
        model_name (str): The name of the GPT model.
        tokenizer: An instance of the Tokenizer class.
        total_token_count (int): The total number of tokens.
        persist_files (bool): Indicates if files are persisted.
        file_ids (list): A list of uploaded file IDs.

    Methods:
        get_token_count(messages: list) -> int: Gets the token count for a list of messages.
        _get_transcript(event_id: int) -> str: Retrieves the transcript for an event.
        find_company_permid(company: str) -> str: Looks up the PERMID for a company.
        get_event_transcripts(company: str, quarter: int=None, year: int=None, start_date: str=None) -> list: 
            Gets event transcripts for a company and specified criteria.
        get_event_transcript(company: str, quarter: int=None, year: int=None) -> list: Gets a single event transcript for a company and specified criteria.
        load_historical_transcripts(company: str, start_date: str='2022-01-01') -> list: Loads historical event transcripts for a company.
        upload_transcript_file(title: str, transcript: str) -> str: Uploads a transcript file.
        begin_conversation() -> list: Starts a conversation with the assistant.
        get_default_event_transcript() -> list: Gets the default event transcript.
        submit_message(message_content: str): Submits a message to the conversation.
        process_messages() -> list: Processes the messages in the conversation.
        _wait_for_run_event(run) -> Run: Waits for the assistant run event to complete.
        close_chat(): Closes the chat and resets the token count.
        _format_openai_messages(messages) -> list: Formats messages for display.
    """    
    def __init__(self, openai_settings: OpenAISettings, aiera_settings, db_settings):
        """
        Initialize a new instance of the class.

        Args:
            openai_settings (OpenAISettings): An instance of the OpenAISettings class containing the OpenAI organization ID and API token.
            aiera_settings: The AIERA settings.
            db_settings: The database settings.

        Attributes:
            client: An instance of the OpenAI class.
            assistant_id: The assistant ID from the OpenAI settings.
            assistant: The assistant retrieved from the OpenAI client.
            thread: A new thread created by the OpenAI client.
            is_verified: A boolean indicating whether the user is verified or not.
            model_name: The name of the model used by the assistant.
            tokenizer: An instance of the Tokenizer class using the assistant model.
            total_token_count: A counter keeping track of the total number of tokens.
            persist_files: A boolean indicating whether to persist files or not.
            aiera_settings: The AIERA settings.
            db_settings: The database settings.
            file_ids: A list to store file IDs.

        Returns:
            None
        """        
        self.client = OpenAI(
            organization = openai_settings.org_id,
            api_key = openai_settings.api_token
        )

        self.assistant_id = openai_settings.assistant_id
        self.assistant = self.client.beta.assistants.retrieve(self.assistant_id)
        self.thread = self.client.beta.threads.create()
        self.is_verified = verify_user(aiera_settings)
        self.model_name = self.assistant.model
        self.tokenizer = Tokenizer(self.assistant.model)
        self.total_token_count = 0
        self.persist_files = openai_settings.persist_files
        self.aiera_settings = aiera_settings
        self.db_settings = db_settings

        # track all files uploaded to openai
        self._file_ids = []
        # track current active file ids
        self._current_file_ids = []

        # remove once using api
        #self.db = AieraReadDatabase(db_config)

    def get_token_count(self, messages: list):
        """
        Get the token count from a list of messages.

        Args:
            self: The object instance.
            messages (list): A list of messages.

        Returns:
            int: The total token count from the input messages.
        """        
        return self.tokenizer.get_token_count(messages)
    
    def _get_transcript(self, event_id: int):

        """
        Get the transcript of a specific event.

        Args:
            event_id (int): The ID of the event.

        Returns:
            str: The transcript of the event.
        """        
        matches = requests.get(f"{self.aiera_settings.base_url}/events/{event_id}/transcript", 
                               headers={"X-API-Key": self.aiera_settings.api_key})


        transcript = ""
        if matches.status_code == 200:
            transcripts = matches.json()

            # now, lets buid
            for transcript_ in transcripts:
                chunks = transcript_["transcript"].split("\n")


                #speaker = self._get_person(transcript.get("person_id"))
                speaker = None
                if speaker:
                    transcript += f"{speaker}: "
                transcript += "\n".join([chunk.strip(" \n") for chunk in chunks])
                transcript += "\n"
                # cleanup missing decimals
                #transcript = transcript.replace("")
                transcript = re.sub(r'(\d) (\d)', r'\1.\2', transcript)

        else:

            logger.error("Unable to get transcript for event %s. Returned %s with reason: %s", event_id, matches.status_code, matches.reason)

        transcript = transcript.replace("<unk>", "")
        return transcript
    
    def find_company_permid(self, company):
        """
        Lookup the permid for the company using the provided db file.
        """
        con =  sqlite3.connect(f"{ROOT_DIR}/aiera_gpt/db/companies.db")
        cur = con.cursor()

        # clean company name
        company = company.replace("   ", " ")
        company = company.replace("  ", " ")

        res = cur.execute(f"SELECT common_name, perm_id FROM companies WHERE LOWER(common_name) LIKE '{company.lower()}%';")
        match = res.fetchone()

        con.close()

        if match is None:
            logger.debug(f"No company match found for {company}")
            return None
        
        else:
            logger.debug("Match found for %s - %s with permid %s", company, match[0], match[1])
            return match[1]



    def get_event_transcripts(self, company: str, quarter: int = None, year: int=None, start_date: str = None):
        """
        Get event transcripts using the provided information.
        """

        if not start_date:
            start_date = f"{year}-01-01"


        # need to compare start date to the quarter
        permid = self.find_company_permid(company)
        if not permid:

            logger.error(f"Unable to find a company match for {company}.")

        params_string = f"event_type=earnings&with_transcripts=1&permid={permid}&start_date={start_date}"

        matches = requests.get(f"{self.aiera_settings.base_url}/events?{params_string}", 
                               headers={"X-API-Key": self.aiera_settings.api_key})
        

        found_event_transcripts = []
        matched_events = []
        if matches.status_code == 200:
            event_matches = matches.json()

            # now we search for matches in the window
            for event in event_matches:

                matched = True

                if quarter and quarter != event["fiscal_quarter"]:
                    matched = False

                if year and year != event["fiscal_year"]:
                    matched = False

                if event["primary_equity"] and matched:
                    matched_events.append(event)

            for match in matched_events:
                transcript = self._get_transcript(match["event_id"])

                # give the item a title for reference
                transcript = f"#{match['title']}\n##Event Date: {match['event_date'].split('T')[0]}\n##Fiscal Quarter: {match['fiscal_quarter']}\n##Fiscal Year: {match['fiscal_year']}\n##Transcript:\n{transcript}"

                file_id = self.upload_transcript_file(match['title'], transcript)
                found_event_transcripts.append(file_id)

        else:

            logger.error("Unable to get events for %s, quarter %s, year %s. Returned %s with reason: %s", company, quarter, year, matches.status_code, matches.reason)

        return found_event_transcripts


    
    def upload_event_transcript(self, company: str, quarter: int = None, year: int = None) -> list:
        """
        Args:
            None
        """
        logger.debug("Getting event transcript for %s, %s, %s", company, quarter, year)
        found_event_transcripts = self.get_event_transcripts(company, quarter, year)

        return found_event_transcripts
    

    def load_historical_transcripts(self, company: str, start_date: str = "2022-01-01") -> list:
        """
        Load historical transcripts for a given company.

        Args:
            self: The object containing the method.
            company (str): The name of the company.
            start_date (str, optional): The start date for the historical data in the format 'yyyy-mm-dd'. Defaults to '2022-01-01'.

        Returns:
            List: A list of file ids uploaded
        """        
        found_event_transcripts = self.get_event_transcripts(company, start_date=start_date)

        logger.debug("Completed loading historical transcripts.")

        return found_event_transcripts
    
    
    def upload_transcript_file(self, title, transcript, sleep=5):
        """
        Upload a transcript file to the storage system.

        Args:
            title (str): The title of the transcript.
            transcript (str): The content of the transcript.

        Returns:
            str: The filename of the uploaded file.

        Raises:
            Exception: If there is an error during the upload process.
        """        

        # create temporary local file
        filename = f"{title.replace(' ', '_').lower()}.md"
        with open(filename, "w") as f:
            f.write(transcript)

        #upload a file with an assistants purpose
        try:
            file = self.client.files.create(
                file = open(filename, "rb"),
                purpose = "assistants"
            )
            self._file_ids.append(file.id)

        except Exception as e:
            logger.error(str(e))

        # optional param to give openai time to index
        if sleep:
            time.sleep(sleep)

        # remove local file 
        #os.remove(filename)
        return file.id


    def begin_conversation(self):
        """
        Begin a conversation by sending a 'hello' message and processing messages.

        Returns:
            str: The response to the 'hello' message.
        """        
        self.submit_message("hello")
        return self.process_messages()



    def get_default_event_transcript(self):
        """
        Get the default event transcript.

        This function creates a system message and returns a list containing the transcript.

        Returns:
            list: The transcript, represented as a list of dictionaries.

        Raises:
            None
        """        
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
        """
        Submit a message to the message thread.

        Args:
            message_content (str): The content of the message.

        Raises:
            None.

        Returns:
            None.
        """        
        logger.debug("Adding message from user to the message thread. Will reference the file_ids: %s", ", ".join(list(set(self._current_file_ids))))
        self.client.beta.threads.messages.create(
            thread_id = self.thread.id,
            role = "user",
            content = message_content,
            file_ids = self._current_file_ids
        )
        

    def process_messages(self) -> List[dict]:
        """ WITH cancel
        Main workflow for processing messages with OpenAI endpoints.
        """
        messages = list(self.client.beta.threads.messages.list(thread_id=self.thread.id))
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

            for each_tool in tools_to_call:
                function_name = each_tool.function.name
                function_arg = each_tool.function.arguments
                if function_arg is not None:
                    function_arg = json.loads(function_arg)

                logger.debug("Will attempt to run %s with args %s", function_name, json.dumps(function_arg))

                if function_name == "upload_event_transcript":
                    file_ids = self.upload_event_transcript(
                        function_arg["company"],
                        quarter = function_arg["quarter"],
                        year = function_arg["year"]
                    )
                    self._current_file_ids = file_ids

                    # At present, file submission is not supported in the 

                    self.client.beta.threads.runs.cancel(
                        thread_id = self.thread.id,
                        run_id = run.id
                    )

                    # Attempt update of files
                    self.client.beta.threads.messages.create(
                        thread_id = self.thread.id,
                        role = "user",
                        content = "",
                        file_ids = file_ids
                    )

                elif function_name == "load_historical_transcripts":
                    file_ids = self.load_historical_transcripts(
                        function_arg["company"],
                        function_arg["start_date"]
                    )

                    self._current_file_ids = file_ids

                    self.client.beta.threads.runs.cancel(
                        thread_id = self.thread.id,
                        run_id = run.id
                    )

                    # Attempt update of files
                    self.client.beta.threads.messages.create(
                        thread_id = self.thread.id,
                        role = "user",
                        content = "",
                        file_ids = file_ids
                    )
                
            
            return self.process_messages()

        if run.status == "completed":
            logger.debug("Completed run.")
            messages = self.client.beta.threads.messages.list(
                thread_id = self.thread.id
            )

            formatted_messages = self._format_openai_messages(messages)
            logger.debug("Current messages:\n%s", json.dumps(formatted_messages))

            self.total_token_count += self.get_token_count(formatted_messages)

            return formatted_messages
        
        else:
            logger.error("Something went wrong. Run status : %s", run.status)


    def _wait_for_run_event(self, run):
        """
        Waits for the completion of an assistant run and returns the final run status.

        Args:
            run (object): The assistant run object.

        Returns:
            object: The final assistant run object with updated status.

        Raises:
            None.
        """        
        logger.debug("Monitoring status of assistant run.")
        i = 0
        while run.status not in ["completed", "failed", "requires_action"]:
            logger.debug("Assistant status is %s. Waiting...", run.status)
            if i > 0:
                time.sleep(10)

            run = self.client.beta.threads.runs.retrieve(
                thread_id = self.thread.id,
                run_id = run.id
            )
            i += 1

        return run


    def close_chat(self):
        """
        Close the chat and reset the total token count.

        This function deletes the thread associated with the chat and sets the total token count to zero.

        Args:
            None

        Returns:
            None
        """        
        self.client.beta.threads.delete(self.thread.id)
        self.total_token_count = 0

    def _format_openai_messages(self, messages):
        """
        Format messages for streamlit.
        """

        new_messages = []
        for message in messages:
            if isinstance(message, ThreadMessage):

                annotations = []

                if message.content[0].text.annotations:

                    for annotation in message.content[0].text.annotations:
                        
                        if isinstance(annotation, TextAnnotationFileCitation):
                            quote = annotation.file_citation.quote
                            quote = quote.replace(" $", " \\$")
                            annotations.append({"quote": quote, "text": annotation.text})
                        else: 
                            print(annotation)
                        #if (file_citation := getattr(annotation, 'file_citation', None)):
                        #        citations.append(f'[{index}] {file_citation.quote} from {cited_file.filename}')

                     #   print(annotation)
                        #annotations.append({"marker": annotation.file_citation.text, 
                        #                    "file_id": annotation.file_citation.file_id,
                        #                    "quote": annotation.file_citation.quote})

                content = message.content[0].text.value
                # add escape so this doesn't render like math
                content = content.replace(" $", " \\$")

                new_message = {"role": message.role, "content": content, "annotations": annotations}
                new_messages.append(new_message)

            else:
                logger.error("Unsupported message type found %s", type(message))

        return new_messages
    
    def __del__(self):
        """
        Delete files uploaded to OpenAI.

        This method is called when the object is being destroyed.

        Raises:
            None
        """        
        if self._file_ids:
            if self.persist_files:
                logger.info("Files are being persisted.")

            if not self.persist_files:
                logger.info("Configured to remove files uploaded to openai. Deleting...")
                for file_id in list(set(self._file_ids)):
                    logger.debug("Removing file %s...", file_id)
                    res = self.client.files.delete(
                        file_id = file_id
                    )
