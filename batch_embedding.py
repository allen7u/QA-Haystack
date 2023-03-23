





import os

batch_path = 'batch/'
# change cwd to batch_path
os.chdir(batch_path)

for dir in os.listdir():
    if not os.path.isdir(dir):
        continue
    
    print(dir)
    from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http

    doc_dir = dir + '/'

    doc = convert_files_to_docs(dir_path=doc_dir, split_paragraphs=False)
    # print(doc)

    from haystack.nodes import PreProcessor

    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length= 100,
        split_respect_sentence_boundary=True,
        split_overlap= 20,
        add_page_number=True
    )
    docs = processor.process( doc )
    print(docs[0])
    # print(docs)

    from haystack.document_stores import FAISSDocumentStore

    try:
        document_store = FAISSDocumentStore.load(index_path= doc_dir +"my_faiss_index.faiss")
    except:
        document_store = FAISSDocumentStore(sql_url="sqlite:///"+ doc_dir +"my_db.db", faiss_index_factory_str="Flat")

    document_store.write_documents(docs)
    from haystack.nodes import EmbeddingRetriever

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_format="sentence_transformers",
    )

    document_store.update_embeddings(retriever)

    # document_store.save("my_document_store")
    document_store.save(doc_dir+"my_faiss_index.faiss")