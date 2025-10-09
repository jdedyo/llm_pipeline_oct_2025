from SETTINGS import *
from sentence_transformers import SentenceTransformer, util

def load_rag_model():
    return SentenceTransformer(RAG_MODEL_ID, cache_folder=RAG_MODEL_DIR)

def prep_plan_data_for_rag(model, all_plan_ids, all_snips, all_formulas, all_years):

    # Calculate the embeddings for the simple plans and complicated plans separately.
    corpus_embeddings=model.encode(all_snips, convert_to_tensor=True)

    plans_data = {'plan_ids': all_plan_ids,
                         'all_snippets': all_snips, 
                         'all_years': all_years,
                         'all_formulas': all_formulas,
                         'all_embeddings': corpus_embeddings
                         }
    return plans_data

# pick up here
def make_query(query_snippet, query_plan_id, query_year):
    return {'snippet':query_snippet, 'plan_id': query_plan_id, 'year': query_year}

def rag_generator(model, query_data, prompt_file, plans_data):

    all_plan_ids = plans_data['plan_ids']
    all_plans = plans_data['all_snippets']
    all_years = plans_data['all_years']
    all_formulas = plans_data['all_formulas']
    corpus_embeddings = plans_data['all_embeddings']

    query_snippet = query_data['snippet']
    query_plan_id = query_data['plan_id']
    query_year = query_data['year']

    # Calculate embedding for the plan in question.
    query_embedding=model.encode(query_snippet, convert_to_tensor=True)

    # Do semantic search for the plan in question in both simple and complicated plans.
    all_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=50)

    # FIXME: EXPLAIN
    all_hits = all_hits[0]
        
    all_hit_indices=[]
    all_hit_plan_ids=[]
    all_hit_snippets=[]
    all_hit_tables=[]
    for hit in all_hits:
        i = hit['corpus_id']
        # Make sure this hit is not identical to the original plan in year and language
        if all_plan_ids[i] != query_plan_id and all_plan_ids[i] not in all_hit_plan_ids:
            all_hit_indices.append(i)
            all_hit_plan_ids.append(all_plan_ids[i])
            all_hit_snippets.append(str(all_plans[all_hit_indices[i]]))
            all_hit_tables.append(str(all_formulas[all_hit_indices[i]]).strip())
        if len(all_hit_indices) >= 20:
            break
    
    snips = all_hit_snippets[:NUM_RAG_EXAMPLES]
    tables = all_hit_tables[:NUM_RAG_EXAMPLES]
    plan_ids = all_hit_plan_ids[:NUM_RAG_EXAMPLES]
    
    return snips, tables, plan_ids

    # print(query_plan_id)
    # print(all_hit_plan_ids)
    # print(all_hit_indices)
    # print(all_plan_ids[all_hit_indices[0]])
    
    # num_examp = 5

    # # Read in the prompt
    # with open(prompt_file, 'r') as file4:
    #     prompt = file4.read()

    # # Fill in the prompt with the snippet to be encoded
    # p = prompt.replace("[YYYY]", str(query_year))
    # p = p.replace("[DOCUMENT]", str(query_doc))

    # for i in range(num_examp):
    #     p = p.replace("[YYYY" + str(i+1) + ']', str(all_years[all_hit_indices[i]]))
    #     p = p.replace("[EXAMPLE" + str(i+1) + ']', str(all_plans[all_hit_indices[i]]))
    #     p = p.replace("[TABLE" + str(i+1) + ']', str(all_formulas[all_hit_indices[i]]).strip())

    
    # return p