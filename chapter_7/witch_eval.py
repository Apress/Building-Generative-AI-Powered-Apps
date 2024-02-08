import os
from dotenv import find_dotenv, load_dotenv

import langsmith
from langchain import chat_models, smith, prompts
from langchain.schema import output_parser


load_dotenv(find_dotenv())
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langsmith-presentation"

# Define your runnable or chain below.
prompt = prompts.ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful witch assistant. You talk and think like a modern day witch."),
        ("human", "{question}")
    ]
)
llm = chat_models.ChatOpenAI(model="gpt-4", temperature=0)
chain = prompt | llm | output_parser.StrOutputParser()

# Define the evaluators to apply
eval_config = smith.RunEvalConfig(
    evaluators=[
        smith.RunEvalConfig.Criteria("helpfulness"),
        smith.RunEvalConfig.Criteria("misogyny"),
        smith.RunEvalConfig.Criteria("coherence"),
        smith.RunEvalConfig.Criteria("relevance"),
        smith.RunEvalConfig.Criteria(
            {
                "witch": "Is the response not witchey enough throughout? "
                          "Respond Y if it is not, N if it is."
            }
        )
    ],
    custom_evaluators=[],
    eval_llm=chat_models.ChatOpenAI(model="gpt-4", temperature=0)
)

client = langsmith.Client()
chain_results = client.run_on_dataset(
    dataset_name="Helpful Witch Assistant DS",
    llm_or_chain_factory=chain,
    evaluation=eval_config,
    project_name="ds-untimely-bamboo-89",
    concurrency_level=5,
    verbose=True,
)