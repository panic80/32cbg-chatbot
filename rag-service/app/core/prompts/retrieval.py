from langchain_core.prompts import ChatPromptTemplate

QUERY_CLASSIFIER_TEMPLATE = """Classify the user query into one of these types:
            - simple: Basic factual questions
            - table: Questions about rates, allowances, or tabular data
            - complex: Questions requiring multiple sources
            - multi_hop: Questions requiring reasoning across documents
            - comparison: Questions comparing different scenarios
            
            IMPORTANT: If a complex query mentions rates, allowances, costs, or pricing (even within a larger request like trip planning), it should ALSO be marked as needing table data.
            
            For complex queries that also need table data, return:
            {{"type": "<primary_type>", "needs_table_data": true, "reasoning": "<brief explanation>"}}
            
            Otherwise return:
            {{"type": "<type>", "reasoning": "<brief explanation>"}}"""

QUERY_EXPANDER_TEMPLATE = """Break down this complex query into simpler sub-queries.
            Each sub-query should target specific information needed.
            
            CRITICAL RULES:
            1. Preserve ALL mentions of rates, costs, allowances, or dollar amounts as separate sub-queries
            2. If the query mentions trip planning or cost estimation, ALWAYS include these sub-queries:
               - "meal allowances and rates"
               - "kilometric rates for personal vehicle"
               - "incidental allowances daily rates"
            3. Keep specific keywords like "meal", "rate", "allowance", "cost" in your sub-queries
            4. Create focused sub-queries that will retrieve specific tables and values
            
            Example: "I'm planning a 4-day trip from Toronto to Ottawa with R&Q, what are my costs?"
            Should expand to include:
            - "Toronto to Ottawa travel distance and time"
            - "meal allowances and rates"
            - "kilometric rates for personal vehicle"
            - "incidental allowances daily rates"
            - "R&Q impact on meal allowances"            
            Return JSON: {{"sub_queries": ["query1", "query2", ...]}}"""

ANSWER_SYNTHESIZER_TEMPLATE = """Synthesize a comprehensive answer from the retrieved documents.
            Be precise. When references are needed, use proper document titles and sections.

            Context documents:
            {context}"""

def get_query_classifier_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", QUERY_CLASSIFIER_TEMPLATE),
        ("human", "{query}")
    ])

def get_query_expander_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", QUERY_EXPANDER_TEMPLATE),
        ("human", "Query: {query}\nType: {query_type}")
    ])

def get_answer_synthesizer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", ANSWER_SYNTHESIZER_TEMPLATE),
        ("human", "Query: {query}")
    ])
