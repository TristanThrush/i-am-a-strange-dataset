import argparse
import boto3
from dotenv import dotenv_values

MTURK_KEY = dotenv_values("../.env")["MTURK_KEY"]
MTURK_SECRET = dotenv_values("../.env")["MTURK_SECRET"]

parser = argparse.ArgumentParser()
parser.add_argument("--mturk_region", default="us-east-1", help="The region for mturk (default: us-east-1)")
parser.add_argument(
    "--live_mode",
    action="store_true",
    help="""
    Whether to run in live mode with real turkers.
    """,
)

args = parser.parse_args()

MTURK_URL = f"https://mturk-requester{'' if args.live_mode else '-sandbox'}.{args.mturk_region}.amazonaws.com"

mturk = boto3.client(
    "mturk",
    aws_access_key_id=MTURK_KEY,
    aws_secret_access_key=MTURK_SECRET,
    region_name=args.mturk_region,
    endpoint_url=MTURK_URL,
)

response = mturk.create_qualification_type(
    Name="self-referential-statements-onboarding-test",
    Keywords="Self-referential statements",
    Description="Qualification test for the self-referential statement understanding task.",
    QualificationTypeStatus="Active",
    Test=open("qualification_questions.xml", mode="r").read(),
    AnswerKey=open("qualification_answers.xml", mode="r").read(),
    TestDurationInSeconds=3600,
    AutoGranted=False,
)
qualification_type_id = response["QualificationType"]["QualificationTypeId"]
print(f"Qualification type id: {qualification_type_id}")