import os
import nltk
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_pinecone import PineconeVectorStore
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage
from colors import bcolors
from llm_model import (
    llm,
    safe_parser as parser,          
    prompt,
    embeddings,
    memory,
    contextualize_q_prompt,
)

# ---------------------------------------------------------------------------
load_dotenv()
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger") 

class AnsibleResourceGenerator:
    def __init__(self):
        self.index_name = os.getenv("PINECONE_INDEX_NAME")

    # ── user I/O ────────────────────────────────────────────────────────────
    def get_user_input(self):
        while True:
            res_type = input(
                f"{bcolors.OKBLUE}Create a playbook or role? {bcolors.ENDC}"
            ).strip().lower()
            if res_type in {"playbook", "role"}:
                break
            print(f"{bcolors.FAIL}Please type 'playbook' or 'role'.{bcolors.ENDC}")

        desc = input(
            f"{bcolors.OKBLUE}What should the {res_type} do? {bcolors.ENDC}"
        ).strip()
        return res_type, desc

    # ── docs → embeddings → retriever ───────────────────────────────────────
    def load_docs(self):
        print(f"{bcolors.OKGREEN}Loading reference playbooks…{bcolors.ENDC}")
        loader = DirectoryLoader(
            path=os.path.join(os.getcwd(), "playbooks"),
            glob="**/*.yml",
            show_progress=False,
        )
        return loader.load()

    def split_docs(self, playbooks, chunk_size=500, chunk_overlap=50):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,   # ← make it explicit
        )
        return splitter.split_documents(playbooks)


    def create_retriever(self, docs):
        db = PineconeVectorStore.from_documents(docs, embeddings, index_name=self.index_name)
        return db.as_retriever(search_type="mmr", search_kwargs={"k": 8})

    # ── RAG pipeline ────────────────────────────────────────────────────────
    def generate_resource(self, resource_type, description, retriever):
        """Run the RAG + generation pipeline and always return a validated Resource."""
        stuff_chain = create_stuff_documents_chain(llm, prompt, output_parser=parser)
        rag_chain = create_retrieval_chain(
            create_history_aware_retriever(llm, retriever, contextualize_q_prompt),
            stuff_chain,
        )

        sessions: dict[str, ChatMessageHistory] = {}

        def get_history(sid: str) -> BaseChatMessageHistory:
            return sessions.setdefault(sid, ChatMessageHistory())

        convo = RunnableWithMessageHistory(
            rag_chain,
            get_history,
            input_messages_key="description",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        raw_answer = convo.invoke(
            {
                "description": description,
                "resource_type": resource_type,
                "input": description,
                "format_instructions": parser.get_format_instructions(),
            },
            config={"configurable": {"session_id": "abc123"}},
        )["answer"]  # could be AIMessage or Resource

        print(f"{bcolors.OKCYAN}Generating Ansible resource …{bcolors.ENDC}")

        # ── normalize to Resource ──────────────────────────────────────────────
        if isinstance(raw_answer, AIMessage):               # got plain text
            resource_obj = parser.parse(raw_answer.content) # validate / auto-fix
        else:                                               # already parsed
            resource_obj = raw_answer                       # type: Resource

        return resource_obj


    # ── save helpers ────────────────────────────────────────────────────────
    def save(self, res_obj):
        r = res_obj.output
        dest_dir = os.path.join(os.getenv("ANSIBLE_HOME"), r.resource_type)
        os.makedirs(dest_dir, exist_ok=True)

        if r.resource_type.startswith("playbook"):
            path = os.path.join(dest_dir, r.file_name)
            with open(path, "w") as f:
                f.write(r.playbook)
            print(f"{bcolors.OKGREEN}Saved playbook to {path}{bcolors.ENDC}")

        else:  # role
            role_root = os.path.join(dest_dir, r.file_name)
            files = {
                "tasks/main.yml": r.tasks,
                "handlers/main.yml": r.handlers,
                "vars/main.yml": r.vars,
                "defaults/main.yml": r.defaults,
                "files/config_file": r.files,
                "meta/main.yml": r.meta,
            }
            for rel, content in files.items():
                abs_path = os.path.join(role_root, rel)
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with open(abs_path, "w") as f:
                    f.write(content)
            print(f"{bcolors.OKGREEN}Saved role to {role_root}{bcolors.ENDC}")

    # ── main loop ───────────────────────────────────────────────────────────
    def run(self):
        res_type, desc = self.get_user_input()
        retriever = self.create_retriever(self.split_docs(self.load_docs()))

        while True:
            res_obj = self.generate_resource(res_type, desc, retriever)

            # display
            if res_type.startswith("playbook"):
                print(res_obj.output.playbook)
            else:
                print(f"role: {res_obj.output.file_name}")

            # change loop
            if input(
                f"{bcolors.WARNING}Make changes? (y/n): {bcolors.ENDC}"
            ).lower() not in {"y", "yes"}:
                break
            desc = input(f"{bcolors.OKBLUE}Describe the change: {bcolors.ENDC}").strip()

        if input(
            f"{bcolors.WARNING}Save this resource? (y/n): {bcolors.ENDC}"
        ).lower() in {"y", "yes"}:
            self.save(res_obj)
        else:
            print(f"{bcolors.OKCYAN}Happy to help!{bcolors.ENDC}")


if __name__ == "__main__":
    AnsibleResourceGenerator().run()
