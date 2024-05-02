# Spark-NLP-on-Yelp-Dataset
-- Yuanshan Zhang, Ananya Anand, Shubham Mishra, Shizuka Takahashi, Ruiqi Jiang

The Yelp dataset consists of 5 JSON files: business (113.4MB), review (4.98GB), tip (172.2MB), user (3.13GB), and checkin (273.7MB), covering 150,346 businesses, 6,990,280 reviews, 908,915 tips by 1,987,897 users, and aggregated check-ins over time for each of the 131,930 businesses. The purpose of this project is to derive insights from the dataset to help businesses improve their services.

Data source: https://www.yelp.com/dataset/documentation/main

# What I did
**1. Research on third-party libraries**\
Since our data contains 7 million reviews, NLP becomes the most important aspect of our analysis. However, Spark MLlib only offers limited and lower-level NLP features such as tokenizer, StopwordsRemover, etc. Applying higher-level NLP techniques with high efficiency becomes the biggest challenge in our project. After researching third-party libraries, I decided to use Spark NLP, which is an open-source NLP library developed by John Snow Labs. Spark NLP is built on top of Apache Spark and TensorFlow and offers a range of higher-level features such as normalizer, stemmer, lemmatizer, pre-trained language models, LLM embeddings, etc.

![示例图片](images/Spark_NLP_vs_Spark_MLlib.png)

**2. Prepare data and create PySpark cluster in Dataproc**\
First, I download the data from Yelp and upload to the Cloud Storage on GCP(Google Cloud Platform). To use Spark NLP, I configured the cluster using the cloud shell by referring to https://github.com/JohnSnowLabs/spark-nlp/tree/master/python#gcp-dataproc:
```bash
gcloud dataproc clusters create <cluster_name> \
  --region=us-east1 \
  --zone=us-east1-b \
  --image-version=2.1-debian11 \
  --master-machine-type=e2-highmem-4 \
  --worker-machine-type=e2-highmem-2 \
  --master-boot-disk-size=50GB \
  --worker-boot-disk-size=50GB \
  --num-workers=2 \
  --bucket=<bucket_name> \
  --optional-components=JUPYTER \
  --enable-component-gateway \
  --metadata 'PIP_PACKAGES=spark-nlp spark-nlp-display google-cloud-bigquery google-cloud-storage' \
  --initialization-actions gs://goog-dataproc-initialization-actions-us-east1/python/pip-install.sh \
  --properties spark:spark.serializer=org.apache.spark.serializer.KryoSerializer,spark:spark.driver.maxResultSize=0,spark:spark.kryoserializer.buffer.max=2000M,spark:spark.jars.packages=com.johnsnowlabs.nlp:spark-nlp_2.12:5.3.3
```

**3. Topic modeling**\
In general, topic modeling refers to the broad task of assigning topics to unlabeled text documents, and the model I pick is the Latent Dirichlet Allocation(LDA), which is a generative probabilistic model that takes Bag of Words or TF-IDF as input, finds groups of words that appear frequently together across different documents, and output probabilistic distribution of the topics.

Before building the model, I first remove special characters from the text. Then, I filtered the reviews and tips for Starbucks to demonstrate how to generate insights for businesses by topic modeling. After training the model, I extracted 5 terms as the topic for each document and created a word cloud:

![示例图片](images/topic_modeling.png)

We observe that in the review dataset, many reviews are centered around ‘pumpkin spice’ and ‘nitro brew’. So, we would recommend Starbucks pay special attention to the reviews of these two products. In the tip dataset, there are many tips centered around ‘wifi’, ‘park’, ‘free’, and ‘slow’, so we can help Starbucks to develop business strategies such as improving app ordering services and providing free wifi to keep and attract customers.

**4. Sentiment Analysis**\
Precise sentiment analysis plays a crucial part in helping businesses refine their services. However, it is quite a challenge to train a well-functional deep learning model from scratch since we have 7 million reviews and limited computational power as well as hardware. Therefore, using a pre-trained model is a commercial and feasible resolution.

The pre-trained model I picked was 'sentimentdl_use_imdb' from Spark NLP. It is trained on classifying IMDB reviews into negative, positive, and neutral categories using `Universal Sentence Encoder`. The reason for choosing this model is because the length and tone of IBDM reviews are very similar to Yelp reviews. After applying the pre-trained model, I did the following analysis:

First, I want to see how many reviews fall into each category of sentiment:

![示例图片](images/sentiment_reviews.png)
- positive reviews are the most common
- negative reviews are about 1/3 of positive reviews
- neutral reviews are the least common

Now, I want to investigate the relationship between stars and sentiment:

![示例图片](images/stars_positive.png)

Like expected, customers who give positive reviews are likely to give high stars(4 or 5). However, lower stars can still be given to the business since sentiment only represents a general emotional tendency. For instance, let's take a look at the following review:

'If you decide to eat here just be aware it is going to take about 2 hours from beginning to end We have tried it multiple times because I want to like it! I have been to its other locations in NJ and never had a bad experience The food is good but it takes a very long time to come out The waitstaff is very young but usually pleasant We have just had too many experiences where we spent way too long waiting We usually opt for another diner or restaurant on the weekends in order to be done quicker'.

Although there is dissatisfaction (the customer complaining about long waiting time) in the review, the model considered its sentiment as positive since compliments (food is delicious) outweigh dissatisfaction. However, the customer only gave 3 stars to the restaurant due to dissatisfaction. Therefore, although positive reviews often correspond to higher stars, positive reviews with low stars are where valuable insights for service improvements lie.

![示例图片](images/stars_negative.png)

Likewise, negative sentiment, in general, corresponds to low stars

![示例图片](images/stars_neutral.png)

The stars given are more evenly distributed for neutral reviews. From the stars distribution based on sentiment categories, we can verify that the pre-trained model assigns relatively accurate sentiment labels.

**4. Conclusion**\
Our tour with Spark NLP begins with topic modeling, which is a useful tool for businesses to pinpoint review topics, understand what products receive most customers' attention, and identify aspects of services they should improve upon. In our analysis, I used Starbucks as an example, recommending Starbucks to look into reviews of 'pumpkin spice' and 'nitro brew' and understand why these two products get many comments. Moreover, I performed topic modeling on tips for Starbucks, helping Starbucks identify aspects of services such as 'wifi', 'park', 'free', and 'slow' that it should pay more attention to. 

Then, I performed sentiment analysis using a pre-trained model from Spark NLP. The model is trained on classifying IMDB reviews into positive, negative, and neutral. From the result, I observe that positive reviews are mostly accompanied by 4 and 5-star ratings; negative reviews are mostly accompanied by 1 and 2-star ratings; neutral reviews are accompanied by approximately even chances of rating from 1 to 5 stars. Further, I used a review example to illustrate why positive reviews with low stars can be where valuable business insights lie and vice versa.

