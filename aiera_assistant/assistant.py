import time
import re
from typing import List
import json
import os

from openai import OpenAI
from openai.types.beta.threads import ThreadMessage
from aiera_assistant.config import AieraSettings, OpenAISettings
from aiera_assistant.__init__ import ROOT_DIR
import logging
import requests

logger = logging.getLogger("aiera_assistant.assistant")


class AieraAssistant:

    """
    A class representing an Aiera Assistant.

    Attributes:
        openai_settings (OpenAISettings): The settings for the OpenAI API.
        aiera_settings: The settings for the Aiera API.
        client: An instance of the OpenAI class.
        assistant_id (str): The ID of the assistant.
        assistant: The assistant object.
        thread: The thread object.
        persist_files (bool): Indicates if files are persisted.
        file_ids (list): A list of uploaded file IDs.

    Methods:
        get_event_transcripts(company: str, quarter: int=None, year: int=None, start_date: str=None) -> list: 
            Gets event transcripts for a company and specified criteria.
        get_event_transcript(company: str, quarter: int=None, year: int=None) -> list: Gets a single event transcript for a company and specified criteria.
        load_historical_transcripts(company: str, start_date: str='2022-01-01') -> list: Loads historical event transcripts for a company.
        upload_transcript_file(title: str, transcript: str) -> str: Uploads a transcript file.
        begin_conversation() -> list: Starts a conversation with the assistant.
        submit_message(message_content: str): Submits a message to the conversation.
        process_messages() -> list: Processes the messages in the conversation.
        _wait_for_run_event(run) -> Run: Waits for the assistant run event to complete.
        close_chat(): Closes the chat and resets the token count.
        _format_openai_messages(messages) -> list: Formats messages for display.
    """    
    def __init__(self, openai_settings: OpenAISettings, aiera_settings):
        """
        Initialize a new instance of the class.

        Parameters:
            openai_settings (OpenAISettings): An instance of the OpenAISettings class containing the OpenAI organization ID and API token.
            aiera_settings: The AIERA settings.

        Attributes:
            client: An instance of the OpenAI class.
            assistant_id: The assistant ID from the OpenAI settings.
            assistant: The assistant retrieved from the OpenAI client.
            thread: A new thread created by the OpenAI client.
            is_verified: A boolean indicating whether the user is verified or not.
            persist_files: A boolean indicating whether to persist files or not.
            aiera_settings: The AIERA settings.
            file_ids: A list to store file IDs.

        Returns:
            None
        """        
        self.client = OpenAI(
            organization = openai_settings.org_id,
            api_key = openai_settings.api_key
        )

        self.assistant_id = openai_settings.assistant_id
        self.assistant = self.client.beta.assistants.retrieve(self.assistant_id)
        self.thread = self.client.beta.threads.create()
        self.persist_files = openai_settings.persist_files
        self.aiera_settings = aiera_settings

        # track all files uploaded to openai
        self._file_ids = []
        # track current active file ids
        self._current_file_ids = []


    def get_events(self, modified_since: str = None,
                bloomberg_ticker: str = None, 
                event_type: str = None,
                start_date: str = None, 
                end_date: str=None, 
                ):
        
        """
        Returns a list of events based on specified parameters.

        Args:
            modified_since (str, optional): Events modified since this date (YYYY-MM-DD format).
            bloomberg_ticker (str, optional): Events related to this Bloomberg ticker symbol.
            event_type (str, optional): Events of this type.
            start_date (str, optional): Events starting from this date (YYYY-MM-DD format).
            end_date (str, optional): Events ending on this date (YYYY-MM-DD format).

        Returns:
            str: The content of the response in JSON format.
        """        
        logger.debug(f"Getting events: {modified_since}, {bloomberg_ticker}, {event_type}, {start_date}, {end_date}")
        
        param_strings = []
        
        for param, item in {"modified_since": modified_since, 
                            "bloomberg_ticker": bloomberg_ticker, 
                            "event_type": event_type,
                            "start_date": start_date,
                            "end_date": end_date,
                            }.items():
            if item is not None:
                param_strings.append(f"{param}={item}")

        param_string = "&".join(param_strings)
        url = f"{self.aiera_settings.base_url}/events?{param_string}"

        matches = requests.get(url, 
                               headers={"X-API-Key": self.aiera_settings.api_key})
        
        content = json.dumps(matches.json())

        return content
    
    def upload_event_transcripts(self, event_ids: list):

        """
        Upload event transcripts to the server.

        Args:
            event_ids (list): A list of event IDs to upload transcripts for.

        Returns:
            list: A list of file IDs corresponding to the uploaded transcripts.

        Raises:
            None.

        Notes:
            - The function expects the self.aiera_settings.base_url and self.aiera_settings.api_key attributes to be set.
            - The function calls the self.upload_transcript_file method to upload each transcript file.
        """        
        logger.debug("Uploading event transcripts: %s", json.dumps(event_ids))
        
        file_ids = []
        for event_id in event_ids:

            event = requests.get(f"{self.aiera_settings.base_url}/events/{event_id}?transcripts=true", 
                               headers={"X-API-Key": self.aiera_settings.api_key})
            
            event_data = event.json()

            transcripts = [event_item["transcript"] for event_item in event_data["transcripts"]]
            # remove transcripts items
            del event_data["transcripts"]

            event_data["transcript"] = "\n".join(transcripts)

            filename = f'{event_id}.json'

            file_id = self.upload_transcript_file(filename, json.dumps(event_data))
            file_ids.append(file_id)

        return file_ids
    
    
    def upload_transcript_file(self, filename, transcript, sleep=5):
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
        os.remove(filename)
        return file.id


    def begin_conversation(self):
        """
        Begin a conversation by sending a 'hello' message and processing messages.

        Returns:
            str: The response to the 'hello' message.
        """        
        self.submit_message("hello")
        return self.process_messages()


    def submit_message(self, message_content: str):
        """
        Submit a message to the message thread.

        Parameters:
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
        """
        Main workflow for processing messages with OpenAI endpoints.
        """
        messages = list(self.client.beta.threads.messages.list(thread_id=self.thread.id))
        print(messages)
        logger.debug("Current messages in thread:\n%s", json.dumps([{"role": mess.role, "content": mess.content[0].text.value} for mess in messages]))


        logger.debug("Creating a remote run using the thread and assistant.")
        run = self.client.beta.threads.runs.create(
            thread_id = self.thread.id,
            assistant_id = self.assistant.id
        )

        run = self._wait_for_run_event(run)

        while run.status == 'requires_action':
            logger.debug("Action required...")
            tools_to_call = run.required_action.submit_tool_outputs.tool_calls


            for each_tool in tools_to_call:
                tool_call_id = each_tool.id
                function_name = each_tool.function.name
                function_arg = each_tool.function.arguments
                if function_arg is not None:
                    function_arg = json.loads(function_arg)

                logger.debug("Will attempt to run %s with args %s", function_name, json.dumps(function_arg))

                if function_name == "get_events":
                    found_events = self.get_events(
                        **function_arg
                    )

                    # Submit events
                    run = self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id = self.thread.id,
                        run_id = run.id,
                        tool_outputs = [{"tool_call_id": tool_call_id, "output": found_events}]
                    )

                    run = self._wait_for_run_event(run)

                elif function_name == "upload_event_transcripts":
                    file_ids = self.upload_event_transcripts(
                        **function_arg
                    )

                    self._current_file_ids =  file_ids

                    # uploads require cancel because of file handling
                    self.client.beta.threads.runs.cancel(
                        thread_id = self.thread.id,
                        run_id = run.id
                    )

                    # Attempt update of files
                    self.client.beta.threads.messages.create(
                        thread_id = self.thread.id,
                        role = "user",
                        content = "Files uploaded.",
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

    def _format_openai_messages(self, messages):
        """
        Format messages for streamlit.
        """

        new_messages = []
        for message in messages:
            if isinstance(message, ThreadMessage):

                content = message.content[0].text.value
                # add escape so this doesn't render like math
                content = content.replace(" $", " \\$")

                new_message = {"role": message.role, "content": content, "annotations": []}
                new_messages.append(new_message)

            else:
                logger.error("Unsupported message type found %s", type(message))

        return new_messages
    
    def __del__(self):
        """
        Delete files uploaded to OpenAI.

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
