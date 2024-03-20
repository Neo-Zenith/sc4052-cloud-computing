## Steps
1. Make sure you install all the dependencies listed in requirements.txt by running:
    * ```bash
      pip install -r requirements.txt
      ```
      
2. Download all article titles of Wikipedia from [here](https://stackoverflow.com/questions/24474288/how-to-obtain-a-list-of-titles-of-all-wikipedia-articles). Extract the file, rename as article_titles.txt and move to `tools` subdirectory.

3. Create a folder `output` and within the folder a folder named `ranks`, then run `tools/topic_extractor.py` to start scrapping Wikipedia webpage data.

4. Run `Main.py` for the experiments. 

5. Run `Plot_Figure2.py` to plot the figure shown in Figure 2 of the report.