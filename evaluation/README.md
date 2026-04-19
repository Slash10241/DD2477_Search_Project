**Ranking**
Use script getRankings.py to get rankings for each predefined list of queries.
Queries used:
    # News & Politics (10)
    "COVID-19 pandemic response and lockdown measures",
    "2020 US presidential election and voter turnout",
    "Black Lives Matter protests and racial justice",
    "Brexit negotiations and the UK leaving the EU",
    "Trump impeachment trial and Senate acquittal",
    "wildfires in Australia and California 2020",
    "Hong Kong protests and China national security law",
    "refugee crisis and immigration policy debates",
    "rise of populism and nationalist movements in Europe",
    "United Nations climate summit and global cooperation",

    # Science & Technology (10)
    "artificial intelligence and machine learning breakthroughs",
    "mRNA vaccines and the future of medicine",
    "climate change policy and the Paris Agreement",
    "SpaceX Falcon 9 and commercial space travel",
    "5G technology rollout and conspiracy theories",
    "CRISPR gene editing and biotech innovation",
    "quantum computing and its practical applications",
    "deepfake technology and synthetic media risks",
    "privacy concerns and surveillance capitalism",
    "autonomous self-driving cars and the future of transport",

    # Business & Economics (10)
    "remote work and the future of the office",
    "stock market crash and economic recession 2020",
    "gig economy and worker rights",
    "startup fundraising and venture capital",
    "Amazon and big tech monopoly concerns",
    "personal finance and investing for beginners",
    "cryptocurrency Bitcoin surge and institutional adoption",
    "supply chain disruption and global trade",
    "universal basic income and wealth inequality",
    "women in leadership and the gender pay gap",

    # Health & Wellbeing (10)
    "mental health during quarantine and isolation",
    "sleep science and improving sleep quality",
    "mindfulness meditation and stress reduction",
    "addiction recovery and substance abuse",
    "diet culture and body positivity movement",
    "long COVID symptoms and post-viral fatigue",
    "therapy and the stigma around seeking help",
    "exercise science and high intensity interval training",
    "gut health microbiome and its effect on mood",
    "burnout and chronic workplace stress",

    # Culture & Society (10)
    "true crime investigations and criminal psychology",
    "social media addiction and its effects on teenagers",
    "diversity and inclusion in the workplace",
    "cancel culture and free speech debate",
    "parenting advice and raising children",
    "feminism and gender equality in 2020",
    "religion spirituality and finding meaning in life",
    "loneliness epidemic and the decline of community",
    "true history of slavery and its lasting legacy",
    "generational differences between millennials and Gen Z",

    # Sport & Entertainment (10)
    "NBA bubble season and LeBron James",
    "history of hip hop and rap music evolution",
    "Tour de France cycling and doping controversies",
    "NFL quarterback rivalries and Super Bowl predictions",
    "Olympics postponement and athlete mental health",
    "Netflix binge watching and the streaming wars",
    "esports gaming industry growth and professional players",
    "Hollywood diversity Oscars so white debate",
    "stand up comedy and the art of the punchline",
    "football soccer tactics and Premier League analysis"

Results are saved in query_results.txt


**Annotation**
LLM Model used to annotate: 
Claude Sonnet 4.6

Prompt used to annotate the results:
You are a data annotation engineer. I want you to go through each query and return a txt file  with relevance for each query.You need to rank these clips on relevance from 0-3 where 0 is not relevant 1 means just the query is mentioned 2 means there is more detail but is not exhaustive and 3 means topic is exhaustively discussed. the 1st line is show name 2nd is episode name then the next paragraph is 2 min clip transcript. Give the results in the following format and save it in a txt file.
<query>
<Show name>
<episode name>
<relevance>
<\n>

Annotated file: annotated_results.txt

**Evaluate metrics**
Run file evaluate_metrics.py
Results stored in results folder.