import pandas as pd
from data_layer import DataLayer
from agents import RetailAgents
from pathlib import Path

class DummyLLM:
    def chat(self, messages, temperature=0.0, max_tokens=512):
        # Simple deterministic responses for testing
        # If asked to generate SQL, return a safe SELECT
        joined = '\n'.join([m.get('content','') for m in messages])
        if 'translate retail analytics questions into DuckDB SQL' in joined:
            return 'SELECT region, SUM(sales) as total_sales FROM sales GROUP BY region ORDER BY total_sales DESC LIMIT 1;'
        if 'executive retail insights assistant' in joined:
            return 'Sales look healthy with multi-region contributions. West leads in total sales.'
        if 'Result CSV:' in joined:
            return 'North had the highest growth; recommendation: investigate inventory.'
        return 'OK'

def main():
    if Path("sample_sales.csv").exists():
        df = pd.read_csv('sample_sales.csv')
    else:
        df = pd.DataFrame(
            [
                {"date": "2025-01-05", "region": "North", "product": "Widget A", "category": "Widgets", "sales": 1000, "units": 10},
                {"date": "2025-01-06", "region": "West", "product": "Widget B", "category": "Widgets", "sales": 1500, "units": 15},
                {"date": "2025-03-15", "region": "West", "product": "Gadget X", "category": "Gadgets", "sales": 5000, "units": 5},
            ]
        )
    data_layer = DataLayer()
    llm = DummyLLM()
    agents = RetailAgents(llm=llm, data_layer=data_layer)

    print('\n=== Summarization Test ===')
    print(agents.summarize_performance(df))

    print('\n=== Q&A Test ===')
    ans = agents.handle_question(df, 'Which region has the highest sales?')
    print(ans)

if __name__ == '__main__':
    main()
