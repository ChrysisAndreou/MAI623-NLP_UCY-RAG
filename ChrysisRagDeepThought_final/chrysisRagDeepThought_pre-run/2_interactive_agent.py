import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import logging 
import config 
from rag_core import initialize_agent, retrieve_relevant_documents
from together import Together

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# ---- Local Generate Answer Function to allow any question not just about UCY ----
def generate_answer(question: str, context_docs: list[str], client: Together, model_name: str = config.QA_GENERATION_MODEL, max_tokens: int = 350, temperature: float = 0.6, top_p: float = 0.9):
    """
    Generates an answer based on the provided context documents using Together AI.

    Args:
        question (str): The user's question.
        context_docs (list[str]): A list of relevant context strings.
        client (Together): An initialized Together client instance.
        model_name (str): The name of the LLM model to use.
        max_tokens (int): Maximum tokens for the generated answer.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.

    Returns:
        str: The generated answer, or an error message.
    """

    context = "\n\n".join(context_docs)

    # Construct the prompt messages 
    messages = [
        # Modified system prompt to allow general knowledge
        {"role": "system", "content": "You are a helpful assistant named Deep Thought. Answer the user's question. Use the provided Context below if it is relevant to the question. If the context is not relevant or doesn't contain the answer, use your general knowledge."},
        # Modified user prompt to be more general
        {"role": "user", "content": f"""Context:
{context}

Question: {question}

Answer:"""}
    ]

    answer = None
    try:
        logging.info(f"Generating answer with model '{model_name}' via Together AI...")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if response and response.choices:
            generated_message = response.choices[0].message
            if generated_message and generated_message.content:
                answer = generated_message.content.strip()
                logging.info(f"Successfully generated answer.")
            else:
                logging.error(f"API response from '{model_name}' did not contain valid content: {response}")
                answer = f"Sorry, the model '{model_name}' response format was unexpected (no content)."
        else:
            logging.error(f"API call to '{model_name}' returned an unexpected response structure: {response}")
            answer = f"Sorry, received an unexpected response structure from the model '{model_name}'."

    except Exception as e:
        # Catch potential API errors (e.g., key issues, model not found, network errors)
        logging.error(f"API call to model '{model_name}' failed: {e}")
        answer = f"Sorry, I encountered an error trying to generate an answer using model '{model_name}'. Please check API key and model availability."

    return answer

# ---- Main Execution (Interactive Chat) ----

def main():
    index, documents_data, model, client = initialize_agent(
        docs_dir=config.DOCS_DIRECTORY,
        index_file=config.INDEX_FILE,
        embedding_model_name=config.EMBEDDING_MODEL,
        chunk_size=config.CHUNK_SIZE
    )

    # Model for generation in interactive mode
    generation_model_name = config.QA_GENERATION_MODEL  
    logging.info(f"Using model for interactive generation: {generation_model_name}")

    print("\\nWelcome to the Interactive RAG Agent!")
    print("""
I am DeepThought, a supercomputer designed to serve as a RAG system for answering questions about the University of Cyprus. I have been tested on 42 questions and answered them all correctly. I am ready for more!

To clarify, the number 42 is my famous answer, known as the "Answer to the Ultimate Question of Life, The Universe, and Everything" from The Hitchhiker's Guide to the Galaxy by Douglas Adams. In my biography, as Douglas mentions, the actual question remains unknown, making the answer cryptic and humorous. So, keep asking questions until you find the ULTIMATE QUESTION!
(type 'exit' to quit)
""")

    while True:
        try:
            query = input("> ")
        except EOFError:
            print("\nExiting.")
            break
        if query.lower() == "exit":
            print("Exiting.")
            break
        if not query.strip():
            continue

        logging.info(f"Received query: '{query}'")
        print("Retrieving relevant documents...")

        # Use the core retrieval function
        # Pass the index, the list of document dicts, the model, and the query
        top_docs_with_scores = retrieve_relevant_documents(
            index=index,
            documents_data=documents_data, 
            model=model,
            query=query,
            k=config.RETRIEVAL_K  
        )
        # retrieve_relevant_documents returns list[(text, score)]
        top_docs_text = [doc for doc, score in top_docs_with_scores]

        if not top_docs_with_scores:
            print("\nNo relevant documents found.")
            answer = "I couldn't find any relevant documents in my knowledge base to answer your question."
        else:
            print(f"\nTop {len(top_docs_with_scores)} relevant document snippets:")
            for i, (doc, score) in enumerate(top_docs_with_scores, 1):
                # Display snippet of the retrieved text
                print(f"{i}. [Score: {score:.4f}] {doc[:200]}...")
            print("\nGenerating answer...")

            # Use the *local* core generation function
            # Needs the question, the list of retrieved text chunks, and the client
            answer = generate_answer(
                question=query,
                context_docs=top_docs_text, # Pass the list of strings
                client=client,
                model_name=generation_model_name
            )

        print("\n"+"-"*15+" Answer "+"-"*15)
        print(answer)
        print("-"*(30+len(" Answer "))+"\n")
        print("hello, i am deep thought how can i help? (type 'exit' to quit)")

if __name__ == "__main__":
    main()