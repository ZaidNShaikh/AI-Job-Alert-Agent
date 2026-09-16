import requests
import smtplib

from email.message import EmailMessage

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama




ALERT_EMAIL = "receiver_email"
GMAIL_EMAIL = "sender_email"


GMAIL_APP_PASSWORD = "enter_yourkey"




model = ChatOllama(
    model="qwen2.5:3b"
)


@tool
def searchJobs(jobRole: str, location: str):
    """
    Searches for remote jobs matching the requested job role.
    """

    print("\nSearching jobs...")

    url = "https://remotive.com/api/remote-jobs"

    params = {
        "search": jobRole,
        "limit": 10
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        jobs = data.get("jobs", [])

        if not jobs:
            return "No jobs found."

        results = []

        for job in jobs[:10]:

            title = job.get(
                "title",
                "Unknown"
            )

            company = job.get(
                "company_name",
                "Unknown"
            )

            job_location = job.get(
                "candidate_required_location",
                "Remote"
            )

            link = job.get(
                "url",
                ""
            )

            results.append(
                f"""
Job Title: {title}
Company: {company}
Location: {job_location}
Apply Link: {link}
"""
            )

        return "\n--------------------------\n".join(results)

    except requests.exceptions.RequestException as e:

        return f"Job search error: {e}"

    except Exception as e:

        return f"Unexpected error: {e}"




@tool
def sendJobAlert(jobDetails: str):
    """
    Sends selected job results to the registered email.
    """

    print("\nSending email...")

    message = EmailMessage()

    message["Subject"] = "AI Job Alert"
    message["From"] = GMAIL_EMAIL
    message["To"] = ALERT_EMAIL

    message.set_content(
        "AI JOB ALERT\n"
        "==============================\n\n"
        "JobBot found the following suitable jobs:\n\n"
        + str(jobDetails)
        + "\n\n"
        "This email was generated automatically by JobBot."
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

            server.send_message(message)

        return "Email sent successfully."

    except Exception as e:

        return f"Email error: {e}"



tools = [
    searchJobs,
    sendJobAlert
]




system_prompt = """
You are JobBot, a simple AI Job Alert Assistant.

Your job is to help the user find relevant jobs.

When the user asks for jobs:

1. Identify the job role.
2. Identify the location if provided.
3. Use the searchJobs tool.
4. Review the returned jobs.
5. Select only jobs relevant to the requested role.
6. Prefer jobs matching the requested location when location
   information is available.
7. Keep the final list short and useful.
8. Use sendJobAlert to email the selected jobs.
9. Do not ask for the email address.
10. After the email is sent, give a short confirmation.

IMPORTANT:
- Do not invent jobs.
- Do not invent companies.
- Do not invent application links.
- Use only jobs returned by the searchJobs tool.
- Keep the terminal response short.
"""




agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt
)



print("=" * 50)
print("        AI JOB ALERT AGENT")
print("=" * 50)

print("Type your job request.")
print("Example:")
print("Find Java developer jobs in Pune")
print("Type 'exit' to stop.")

while True:

    query = input("\nYou: ")

    if query.lower() in ["exit", "quit"]:

        print("\nJobBot: Goodbye!")
        break

    try:

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            }
        )

        final_message = result["messages"][-1]

        print("\nJobBot:")
        print(final_message.content)

    except Exception as e:

        print("\nJobBot: Something went wrong.")
        print("Error:", e)