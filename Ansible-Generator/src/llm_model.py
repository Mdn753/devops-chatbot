from __future__ import annotations
import os
from typing import Union
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import YamlOutputParser
from langchain.output_parsers import OutputFixingParser  # ← NEW
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from template import template, contextualize_q_system_prompt

# =============================================================================
# 1.  Environment
# =============================================================================
load_dotenv()

# =============================================================================
# 2.  YAML ↔ Pydantic schema
# =============================================================================
class PlaybookResource(BaseModel):
    resource_type: str = Field(description="must be 'playbooks'")
    file_name: str
    playbook: str


class RoleResource(BaseModel):
    resource_type: str = Field(description="must be 'roles'")
    file_name: str
    tasks: str
    handlers: str
    vars: str
    defaults: str
    files: str
    meta: str


class Resource(BaseModel):
    output: Union[PlaybookResource, RoleResource]


# =============================================================================
#  Chat LLM – Ollama default
# =============================================================================
model_type = os.getenv("CHAT_MODEL_TYPE", "ollama://llama3")

if model_type.startswith("gpt"):
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model_type, temperature=0)
    print(f"[INFO] Using OpenAI model: {model_type}")

elif model_type.startswith("claude"):
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model=model_type, temperature=0)
    print(f"[INFO] Using Anthropic model: {model_type}")

elif model_type.startswith("ollama://"):
    from langchain_community.llms.ollama import Ollama

    ollama_model = model_type.split("://", 1)[1] or "llama3"
    llm = Ollama(model=ollama_model, temperature=0)
    print(f"[INFO] Using local Ollama model: {ollama_model}")

else:
    raise ValueError(
        "Set CHAT_MODEL_TYPE to 'gpt-*', 'claude-*', or 'ollama://model'"
    )

# =============================================================================
# 4.  Prompt plumbing
# =============================================================================
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(template),
        ("placeholder", "{chat_history}"),
        HumanMessagePromptTemplate.from_template(
            "Generate an Ansible {resource_type}.\n"
            "User request: {description}\n\n"
            "**Reply ONLY with YAML that matches the schema below, and wrap the "
            "entire YAML under a top-level key called `output`:**\n"
            "{format_instructions}"
        ),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


parser = YamlOutputParser(pydantic_object=Resource)
safe_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

# =============================================================================
# 5.  Free Hugging Face embeddings
# =============================================================================
EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", "nomic-ai/nomic-embed-text-v1")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL, model_kwargs={"trust_remote_code": True}
)
embed_dimension = len(embeddings.embed_query("hello world"))

# =============================================================================
# 6.  Pinecone index (auto-create / sync dim)
# =============================================================================
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "ansible-beam")

idx = next((i for i in pc.list_indexes() if i.name == index_name), None)
if idx and idx.dimension != embed_dimension:
    print(f"[INFO] Recreating Pinecone index (dim {idx.dimension} → {embed_dimension})")
    pc.delete_index(index_name)
    idx = None

if not idx:
    pc.create_index(
        name=index_name,
        dimension=embed_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"[INFO] Created Pinecone index '{index_name}' ({embed_dimension} dims)")
