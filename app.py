import os # to access environment variables like api keys

from dotenv import load_dotenv # to load env variables from .env files(api keys used)

from langchain_community.document_loaders import DirectoryLoader, UnstructuredPowerPointLoader

from langchain_groq import ChatGroq # using groq for its generous free tier
# tried using gemini but the tokens provided were too low

from langchain_huggingface import HuggingFaceEmbeddings

import glob # to get list of files in a directory for loading multiple documents at once

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredPowerPointLoader # to load different types of documents

from langchain_community.document_loaders import Docx2txtLoader # to load docs

from langchain_text_splitters import RecursiveCharacterTextSplitter # to split pdf info into chunks

from langchain_community.vectorstores import FAISS # to create a veector store for retrieval of documnet/pdf info

from langchain_core.prompts import ChatPromptTemplate # creating a prompt template for RAG chain

from langchain_core.runnables import RunnablePassthrough # to format retrieved info from docs and return it as context for the prompt

from langchain_core.output_parsers import StrOutputParser # to parse op from llm and return it as a string for the final answer

# step 0:  loading api key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # for google search tool, not used yet
# will be added in the future to allow users to ask questions that aren't in the document and use google search to find the answer

# step 1:  loading pdf(sample.pdf here)
# will add a file uploader in the future to allow user uploads
# only supports pdf for now
# will add support for more file types in the future
print("Loading document...")

# loader = PyPDFLoader("sample.pdf") # change later

# using the files from documents folder to load multiple types of documents -- pdf, doc, txt, pptx
# loading multiple documents and combining them into one loader to create a more robust academic assistant that can answer questions from multiple sources

# loader = PyPDFLoader("sample.pdf") [replacing with directory]
# day 2 changes [adding support for other data types(ppt, txt, doc)]
# loader = PyPDFLoader("documents/java_u1.pdf") # loading pdf from directory
# loader = Docx2txtLoader("documents/coa_u3.doc") # loading doc type document 
# loader = UnstructuredPowerPointLoader("documents/dm_u1.pptx") # loading pptx type doc
# loader = TextLoader("documents/se_u1.txt") # loading txt type document
all_docs = [] # creating an empty list to store loaded documents
supported = {
   # using a dictionary to map file extensions to the corresponding loaders
   # to easilyu load diff document types without much change in code
   # aka reusability and scalability!!! like in that one unit i forgot which one
   # in software engineering i think
   # see i use my brain sometimes lolz
   ".pdf": PyPDFLoader,
   ".doc": Docx2txtLoader,
   ".docx": Docx2txtLoader,
   ".pptx": UnstructuredPowerPointLoader,
   ".txt": TextLoader,
}
for filepath in glob.glob("documents/*.*"): 
    # using glob to get a list of all files in the documents directory
    # does glob stand for global? 
    # looked it up apparently it finds pathnames using pattern matching rules similar to the unix shell
    # its syntax is glob.glob(pathname, *, recursive = False)
    # pathname is the pattern to match
    # so here it matches all files in the documents directory with any extension
    # ** means any directories and subdirectories
    # basically all the folders in the documents folder and all the files in those folders
    # the . means any file name
    # * means anyextension
    ext = os.path.splitext(filepath)[1].lower() # getting the file extension to determine which loader to use
    # so ext is the key to the dictionary that maps to the corresponding loader
    # hence the [1] indexing
    if ext in supported:
        print(f"Loading {filepath}...") # printing name of file being loaded
        loader = supported[ext](filepath) # using the corresponding loader to load the doc based on extension
        all_docs.extend(loader.load()) # the empty list we made earlier is being used now!
        # see line 51 (?) (might change whatev) to look back
        # anyway here we load the doc and extend the all_docs list with loaded doc
        # we use extend instead of append bc it might return a list of docs maybe
        # depends on the directory
    else:
        print(f"{filepath} has unsupported filetype, skipping.") # show a message when skipping an unsupported file

# [OLD] docs = loader.load()
print(f"loaded {len(all_docs)} pages.") # printing number of files loaded
# just because :p
docs = all_docs # using list of all loaded docs to get info for the answers
print(f"Loaded {len(docs)} pages")

# step 2: splitting context of loaded pdf into chunks for processing
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50) # using recursive character splitter to split the pdf into chunks of 500 ish characters, with an overlap of 50 characters 
# overlap is used to maintain some context between chunks
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# step 3: creating embeddings for the chunks and storing them in a vector store for retrieval later
print("Creating vector store (may take 30~60 seconds)...")
embeddings = HuggingFaceEmbeddings( # using huggingface embeddings as it has good performance, is free of charge, and easy to use with langchain 
    model_name = "all-MiniLM-L6-v2" # using a small model to speed up the embedding process but can be changed to a larger model for better performance.
    # the larger the model, the better the performance
    # but it is also slower and may be unnecessary for small documents
)
vectorstore = FAISS.from_documents(chunks, embeddings) 
# creating a vector store using FAISS to store the embeddings of the chunks for retrieval later
# FAISS stands for Facebook AI Similarity Search and it is a library for efficient similarity search and clustering of dense vectors
retriever = vectorstore.as_retriever(search_kwargs = {"k" : 3})
# creating a retriever from the vector store to retrieve the most relevant chunks based on cosine similarity
# k is the number of chunks to retrieve
# k can be chsnged based on the size of the document and the desired performance
# kwargs means keyword arguments and it is used to pass additional arguments to the retriever
# in this case, we pass the number of chunks to retrieve as a keyword argument
# the retriever uses this argument to determine how many chunks to retrieve
print("Vector store ready!")

# step 4: setting up llm (groq here) 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model = "llama-3.1-8b-instant", # using a smaller model to speed up response time
    # can be changed to a larger model for better performance later
    api_key = GROQ_API_KEY, # loaded from .env file
    temperature = 0.3 # setting temp to 0.3 to make the responses more focused and less random
    # can be changed to warmer for a more creative response
    # but it increases chances of getting an incorrect answer
    # so we use a lower temp to get a more accurate answer
)

# step5: create prompt template + rag chain
# used prompt engineering (assisted by ai) to create an effective prompt that minimizes token usage while maximizing response quality
# and accuracy by instructing the llm to only use the retrieved context and
# to not make up an answer when the required info isnt in the context
prompt = ChatPromptTemplate.from_template("""
Role: Expert Academic Assistant.
Task: Synthesize "Learning Guides" from provided context (Textbooks, Notes, Question Papers, Labs).

Guidelines:
1. Multi-Perspective: Combine theory (textbooks) with practicals (labs) and exam context (question papers)[cite: 6, 208].
2. Structure: Use headings for Concept, Examples, and Exam Application.
3. Attribution: Tag info with source type (e.g., [Notes] or [Exam 2023])[cite: 212].
4. Precision: Use LaTeX for math/science. Use code blocks for CS topics[cite: 103, 218].
5. Transparency: If info is missing, state: "Context lacks [topic] details"[cite: 212, 220].

Context: {context}
Query: {question}
Result:
""")

# this prompt template is used to instruct the llm to only use the retrieved context to answer the question and to not make up an answer
# if the required info isnt in the context, the llm responds with a default statement

def format_docs(docs): # this function formats the retrieved chunks from the retriever
    # into a string that can be used as context for the prompt template
    return "\n\n".join(doc.page_content for doc in docs)
# it takes the retrieved docs and joins their page content with two newlines in between to create a single string that can be used as context for the prompt template

rag_chain = ( # creating RAG chain using the retriever, prompt template, llm, and op parser
    {"context": retriever | format_docs, "question": RunnablePassthrough()} # this part of the chain retrieves relavent chunks from the vector store based on the question and formats them as context for the prompt template
    | prompt # this part of the chain takes the formatted context and the question and creates a prompt for the llm
    | llm # this part of the chain takes the prompt and generates a response using llm
    | StrOutputParser() # this part of the chain takes the response from the llm and parses it as a string to be returned as the final answer
)

# step6: questions from user + answering
print("Ready to answer! Type 'quit' to exit\n")
while True: # using an infinite loop to keep asking and answering questions until user types quit
    question = input("Ask a question: ") # taking question input from user
    if question.lower() == 'quit': # stop if user types quit
        break
    print("\nThinking...")
    answer = rag_chain.invoke(question) # invoking the ragchain  with question to get answer
    print(f"\nAnswer: {answer}") # print the answer
    print("_" * 50) # printing a line to separate answers
    # for better readability when asking multiple questions