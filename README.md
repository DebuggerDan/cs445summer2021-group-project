# cs445summer2021-group-project
CS 445 Summer 2021 - Group Programming Project


Machine Learning Group Project Proposal


Team Epic: Dan J, Taylor Noah, Camilo Schaser-Hughes

- Proposal: Utilize Gaussian Naive Bayes, Fuzzy C-Means, K-means, and Average results on the MNIST database to determine differential efficiency and accuracy of various machine learning algorithms.







Archive:

- Important Dates: #1. Due by 15th of August, Sunday
- Determine by 9th of August, Monday for data scraping / training data formatization
- Determine by 11th of August, Wednesday for having training data sets formatted & ready to be used
- Determine by 13th of August, Friday for having at least two or more trained NLP models ready to be analyzed -and/or fine tuned

-[Previous Proposal: Train a NLP-based neural network to function as a highly capable PSU helpbot using PSU-specific public resources, e.g., OIT Helpdesk & CAT.pdx.edu. By using, comparing, and contrasting with models such as GPT-2 (354M to 1554M) and the GPT-3 (OpenAI Beta API) and possibly a basic spaCy trained model, we can differentiate efficacy & the alternating training set variations to see which model was most effective in their functionality as a PSU help bot.]


Group Goals: -a. Find reliable & relevant sources of audio-to-text transcripts, captioning, or good contextual PSU stuff (e.g. OIT Helpdesk [Atlassan Q&A site] & the Computer Action Team website oit.pdx.edu)
-B. Contact the DRC of PSU for possibly audiotext transcription sources and/or contact DaBAH of the CAT for helpful text materials for the training data--

AI Training Documentation:
- OpenAI (GPT-3): https://beta.openai.com/docs/guides/answers
- OpenAI Beta API ToS [IMPORTANT]: https://beta.openai.com/policies/terms-of-use

E,g. (.json file formatization-wise for GPT-3/OpenAI Beta API Models)
{"text": "puppy A is happy", "metadata": "emotional state of puppy A"}
{"text": "puppy B is sad", "metadata": "emotional state of puppy B"}
CURL-commandline:
curl https://api.openai.com/v1/files \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F purpose="answers" \
  -F file="@myfile.jsonl"
OpenAI’s Python API Client:
openai.File.create(file=open("myfile.jsonl"), purpose='answers')

AI / NLP Materials currently available:
OpenAI Beta API Access (Dan)
GPT-2 345M and 1554M models (Set-up using tensorflow, Dan)
spaCy model processing (Dan)

