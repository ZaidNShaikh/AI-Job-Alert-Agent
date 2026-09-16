import os
from dotenv import load_dotenv

import requests
import smtplib

from email.message import EmailMessage

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_ollama import ChatOllama
from langchain.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk
)


# ---------------------------------------
# QWEN MODEL
# ---------------------------------------

model = ChatOllama(
    model="qwen2.5:3b"
)



load_dotenv()

ALERT_EMAIL = os.getenv("ALERT_EMAIL")
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")


@tool
def searchJobs(jobRole, location):
    """
    Searches remote jobs for the given job role.
    """

    url = "https://remotive.com/api/remote-jobs"

    params = {
        "search": jobRole,
        "limit": 5
    }

    try:

        print("\nSearching for jobs...\n")

        response = requests.get(
            url,
            params=params
        )

        response.raise_for_status()

        data = response.json()

        jobs = data.get(
            "jobs",
            []
        )

        if not jobs:

            return "No jobs found."

        result = ""

        for job in jobs[:5]:

            title = job.get(
                "title",
                "Unknown"
            )

            company = job.get(
                "company_name",
                "Unknown"
            )

            jobLocation = job.get(
                "candidate_required_location",
                "Remote"
            )

            description = job.get(
                "description",
                "No description"
            )

            link = job.get(
                "url",
                ""
            )

            result += (
                f"Job Title: {title}\n"
                f"Company: {company}\n"
                f"Location: {jobLocation}\n"
                f"Description: {description[:500]}\n"
                f"Apply Link: {link}\n"
                f"--------------------------\n"
            )

        return result

    except Exception as e:

        return f"Search error: {e}"



@tool
def sendJobAlert(jobDetails):
    """
    Automatically sends the job search results
    to the fixed email address.
    """

    jobDetails = str(jobDetails)

    message = EmailMessage()

    message["Subject"] = "AI Job Alert"

    message["From"] = GMAIL_EMAIL

    message["To"] = ALERT_EMAIL

    message.set_content(
    "AI JOB ALERT\n"
    "==============================\n\n"
    "The AI Job Alert Agent found these jobs:\n\n"
    + str(jobDetails)
)

    try:

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as server:

            server.login(
                GMAIL_EMAIL,
                GMAIL_APP_PASSWORD
            )

            server.send_message(
                message
            )

        return "Job alert email sent successfully."

    except Exception as e:

        return f"Email error: {e}"


# ---------------------------------------
# TOOLS
# ---------------------------------------

tools = [
    searchJobs,
    sendJobAlert
]


# ---------------------------------------
# SYSTEM PROMPT
# ---------------------------------------

system_prompt = """
You are an AI Job Alert Assistant.

Your name is JobBot.

Your job is to find suitable jobs for the user.

When the user asks to search for jobs:

1. Use the searchJobs tool.

2. Analyze the jobs returned by the tool.

3. Select jobs that are relevant to the
   user's requested role.

4. Automatically use the sendJobAlert tool
   to send the suitable jobs to the fixed
   registered email address.

5. Do not ask the user for an email address.

6. After sending the email, tell the user
   that the job alert was sent.

Keep your response simple.
"""


# ---------------------------------------
# CREATE AGENT
# ---------------------------------------

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt
)



messages = []


while True:

    query = input(
        "\nEnter your message: "
    )

    if query.lower() == "exit" or query.lower() == "quit":
        break

    messages.append(
        HumanMessage(
            content=query
        )
    )

    aiMessage = ""

    print("\nJobBot: ", end="")

    # -----------------------------------
    # STREAMING
    # -----------------------------------

    for chunk, metadata in agent.stream(
        {
            "messages": messages
        },
        stream_mode="messages"
    ):

        if isinstance(
            chunk,
            AIMessageChunk
        ):

            if isinstance(
                chunk.content,
                str
            ):

                aiMessage += chunk.content

                print(
                    chunk.content,
                    end="",
                    flush=True
                )

            for tool in chunk.tool_call_chunks:

                toolName = tool.get(
                    "name"
                )

                if toolName:

                    print(
                        f"\nCalling {toolName}...",
                        flush=True
                    )

    print()

    messages.append(
        AIMessage(
            content=aiMessage
        )
    )