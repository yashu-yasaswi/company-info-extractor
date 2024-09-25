import re
import os
from langchain import OpenAI, PromptTemplate
from langchain.agents import load_tools, initialize_agent
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set your API keys from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
serpapi_api_key = os.getenv('SERPAPI_API_KEY')

# Initialize LLM
llm = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Load tools
tools = load_tools(["serpapi"], llm=llm, serpapi_api_key=os.environ['SERPAPI_API_KEY'])

# Initialize agent
agent = initialize_agent(tools, agent="zero-shot-react-description", llm=llm)

# Define your PromptTemplate for extended company information
extended_info_template = PromptTemplate.from_template(
    "Retrieve the following information for the company {company_name}:\n"
    "- Contact details and address\n"
    "  - Email\n"
    "  - Phone number\n"
    "  - Website\n"
    "  - Postal code\n"
    "  - Address\n"
    "  - City\n"
    "- Products\n"
    "- Services\n"
    "- Revenue\n"
    "- Competitors\n"
    "- Branches\n"
    "- Careers\n"
    "Please ensure the information is accurate and up-to-date."
    "The details must be formatted as dictionary key-value pairs."
)

# Define validation functions for phone number and email
def validate_phone_number(phone):
    pattern = re.compile(r"^\+?\d{1,3}?\d{10}$")
    return pattern.match(phone)

def validate_email(email):
    pattern = re.compile(r"^[^@]+@[^@]+\.[^@]+$")
    return pattern.match(email)

# Define extraction functions
def extract_email(line):
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", line)
    return match.group(0) if match else None

def extract_phone_number(line):
    match = re.search(r"\+?\d{1,3}?\d{10}", line)
    return match.group(0) if match else None

def extract_key_value(line, key):
    pattern = re.compile(rf"{key} - ([\w\s,./:()]+)")
    match = pattern.search(line)
    return match.group(1) if match else None

# Function to parse the plain text output and convert it to a dictionary
def parse_extended_info_result(result):
    info_dict = {}
    # Extract email
    email = extract_email(result)
    if email and validate_email(email):
        info_dict["Email"] = email

    # Extract phone number
    phone = extract_phone_number(result)
    if phone and validate_phone_number(phone):
        info_dict["Phone Number"] = phone

    # Extract website
    website = extract_key_value(result, "Website")
    if website:
        info_dict["Website"] = website

    # Extract postal code
    postal_code = extract_key_value(result, "Postal code")
    if postal_code:
        info_dict["Postal Code"] = postal_code

    # Extract address
    address = extract_key_value(result, "Address")
    if address:
        info_dict["Address"] = address

    # Extract city
    city = extract_key_value(result, "City")
    if city:
        info_dict["City"] = city

    # Extract additional information
    products = extract_key_value(result, "Products")
    if products:
        info_dict["Products"] = products

    services = extract_key_value(result, "Services")
    if services:
        info_dict["Services"] = services

    revenue = extract_key_value(result, "Revenue")
    if revenue:
        info_dict["Revenue"] = revenue

    competitors = extract_key_value(result, "Competitors")
    if competitors:
        info_dict["Competitors"] = competitors

    branches = extract_key_value(result, "Branches")
    if branches:
        info_dict["Branches"] = branches

    careers = extract_key_value(result, "Careers")
    if careers:
        info_dict["Careers"] = careers

    return info_dict

# Function to run the agent and retrieve company information
def retrieve_company_info(company_name):
    extended_info_prompt = extended_info_template.format(company_name=company_name)
    extended_info_result = agent.run(extended_info_prompt)
    parsed_info = parse_extended_info_result(extended_info_result)
    return parsed_info
