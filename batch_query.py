





import os

batch_path = 'batch/'
# change cwd to batch_path
os.chdir(batch_path)

for dir_ in os.listdir():
    if not os.path.isdir(dir_):
        continue
    
    print(dir_)
    
    doc_dir = dir_ + '/'

    from haystack.document_stores import FAISSDocumentStore

    try:
        document_store = FAISSDocumentStore.load(index_path= doc_dir +"my_faiss_index.faiss")
    except:
        print("No index found, skipping {}...".format(dir_) )
        continue

    from haystack.nodes import EmbeddingRetriever

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_format="sentence_transformers",
    )
    # document_store.update_embeddings(retriever)
    # document_store.save(index_path="my_faiss_index.faiss")
    from haystack.nodes import FARMReader

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
    from haystack.pipelines import ExtractiveQAPipeline

    pipe = ExtractiveQAPipeline(reader, retriever)
    # from haystack.utils import print_answers

    # prediction = pipe.run(
    #     query="how could machine learn?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    # )
    # print_answers(prediction, details="minimum")
    # read questions from file
    with open("../questions.txt", "r") as f:
        queries = f.readlines()
    # queries = [
    #     "how could machine learn?",
    #     "what is the best way to learn machine?",
    #     "what is the worst way?"
    # ]
    from haystack.utils import print_answers
    import json

    # save answers to file
    with open( dir_ + "_answers.txt", "w") as f:       
        for query in queries:
            prediction = pipe.run(query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 2}})
            
            print_answers(prediction, details="minimum")

            query = prediction['query']
            f.write("### " + query)
            # f.write("---------------------------------------\n")
            f.write("\n")
            answers = prediction['answers']
            # print(answers)
            buffer = []
            for i, answer in enumerate(answers):
                # buffer.append("Num: " + str(i))
                # buffer.append(query)
                buffer.append("###### > " + answer.answer)
                buffer.append("" + str(round(answer.score, 3)))
                # buffer.append("-------------------")
                start_idx = answer.offsets_in_context[0].start
                before_span = answer.context[:start_idx]
                end_idx = answer.offsets_in_context[0].end
                after_span = answer.context[end_idx:]
                marked_context = before_span + " __" + answer.context[start_idx:end_idx] + "__" + after_span
                buffer.append("" + marked_context) 
                buffer.append("" + answer.context)
                buffer.append("" +answer.meta['name']+"\n")
                # buffer.append("\n")

            f.write("\n".join(buffer))

            f.write("\n\n")

    # copy the file and rename ext to md 
    import shutil
    shutil.copyfile(dir_ + "_answers.txt", dir_ + "_answers.md")


    