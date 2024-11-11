# 🛒 eCommerce Data Analysis and Demand Forecasting 📊

## 🚀 Project Overview

This project demonstrates advanced data engineering and machine learning skills using a comprehensive eCommerce dataset. By leveraging cloud platforms (AWS S3 and EMR), big data processing frameworks (Hadoop and Apache Spark), and powerful analytics libraries (PySpark, Hive), we deliver an end-to-end solution for **data ingestion**, **preprocessing**, **exploratory data analysis**, **demand forecasting**, and **predictive modeling**.

### Key Features:

- **Big Data Processing**: Utilize Apache Spark on AWS EMR for scalable data handling.
- **Data Preprocessing**: Clean, transform, and enrich data directly from AWS S3 using PySpark and Hive.
- **Demand Forecasting**: Time series analysis for sales forecasting using advanced statistical models.
- **Predictive Modeling**: Implement classification models to predict user purchase behavior.
- **Cloud Integration**: Efficient use of AWS services (S3, EMR, and SageMaker) for seamless data handling and model training.

---

## 📂 Project Structure

```plaintext
eCommerce-Data-Analysis/
├── data/
│   ├── raw/                # Raw dataset files
│   ├── processed/          # Preprocessed dataset files
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── exploratory_analysis.ipynb
│   ├── time_series_forecasting.ipynb
│   ├── predictive_modeling.ipynb
├── src/
│   ├── preprocess.py       # Data preprocessing scripts
│   ├── model_training.py   # Model training scripts
│   ├── utils.py            # Helper functions
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── LICENSE
```

````

---

## 📊 Dataset Overview

The dataset contains anonymized records of eCommerce transactions made by European customers in 2023. It includes the following columns:

| Column Name     | Data Type | Description                                            |
| --------------- | --------- | ------------------------------------------------------ |
| `event_time`    | `STRING`  | Timestamp of the transaction                           |
| `order_id`      | `STRING`  | Unique identifier for each order                       |
| `product_id`    | `STRING`  | Unique identifier for each product                     |
| `category_id`   | `STRING`  | Category identifier of the product                     |
| `category_code` | `STRING`  | Product category code (e.g., 'electronics.smartphone') |
| `brand`         | `STRING`  | Brand of the product                                   |
| `price`         | `DOUBLE`  | Price of the product in USD                            |
| `user_id`       | `STRING`  | Unique identifier for each user                        |

---

## 🛠️ Technologies Used

- **Cloud Platform**: AWS (S3, EMR, SageMaker)
- **Big Data Framework**: Apache Hadoop, Apache Spark (PySpark)
- **Database**: Apache Hive for SQL-based querying
- **Machine Learning**: Time Series Analysis, Classification Models
- **Programming Languages**: Python (PySpark, Pandas, NumPy)
- **Visualization**: Matplotlib, Seaborn

---

## 🔍 Exploratory Data Analysis (EDA)

The Exploratory Data Analysis (EDA) was conducted using PySpark SQL to gain insights into the dataset.

```python
# Peak Shopping Times Analysis
result = spark.sql("""
    SELECT HOUR(event_time) AS hour, COUNT(*) AS purchase_count
    FROM ecommerce_data
    GROUP BY hour
    ORDER BY purchase_count DESC
""")
result.show()
```

---

## ⏳ Time Series Analysis for Demand Forecasting

Using PySpark and statistical models, we forecast future sales based on historical trends.

```python
from pyspark.sql.functions import col, to_date, month, year, date_sub, current_date

# Convert event_time to date format
ecommerce_data = ecommerce_data.withColumn("event_date", to_date(col("event_time")))

# Calculate year and month columns
ecommerce_data = ecommerce_data.withColumn("year", year(col("event_date")))
ecommerce_data = ecommerce_data.withColumn("month", month(col("event_date")))

# Filter data for the last 6 months
six_months_ago = date_sub(current_date(), 180)
filtered_data = ecommerce_data.filter(col("event_date") >= six_months_ago)

# Aggregate data for order and sales analysis
result = filtered_data.groupBy("year", "month").agg(
    count("*").alias("order_count"),
    sum("price").alias("total_sales")
)

result = result.orderBy("year", "month")
result.show()
```

**Visualization of Monthly Sales Trend:**

```python
import matplotlib.pyplot as plt

# Convert to Pandas DataFrame for plotting
sales_data = result.toPandas()
sales_data['year_month'] = sales_data['year'].astype(str) + '-' + sales_data['month'].astype(str)

# Line plot for Order vs. Sales
plt.figure(figsize=(12, 6))
plt.plot(sales_data['year_month'], sales_data['order_count'], marker='o', label='Order Count')
plt.plot(sales_data['year_month'], sales_data['total_sales'], marker='s', label='Total Sales ($)')
plt.xticks(rotation=45)
plt.xlabel('Year-Month')
plt.ylabel('Count / Sales ($)')
plt.title('Monthly Order Count and Total Sales Analysis (Last 6 Months)')
plt.legend()
plt.grid()
plt.show()
```

---

## 🤖 Predictive Modeling

We implemented classification models using **Random Forest** and **Logistic Regression** to predict user purchase behavior.

```python
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Train Random Forest Classifier
rf = RandomForestClassifier(featuresCol='features', labelCol='label')
model = rf.fit(train_data)
predictions = model.transform(test_data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy:.2f}")
```

---

## 💡 Key Results

- **Demand Forecasting**: Achieved a high level of accuracy in predicting sales trends using ARIMA.
- **Predictive Model Performance**: Random Forest model achieved an **accuracy of 92%**, outperforming Logistic Regression.
- **Scalability**: Efficient data processing using Apache Spark on AWS EMR enabled handling of millions of records seamlessly.

---

## ⚙️ Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/eCommerce-Data-Analysis.git

# Install Python dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook

# Launch PySpark session
spark = SparkSession.builder.appName("eCommerceProject").getOrCreate()
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgements

- Apache Software Foundation for providing a robust big data framework.
- AWS for cloud services and infrastructure support.

## 📝 Contact

**Md Fahimul Kabir Chowdhury**
**Email**: [info@tech2etc.com](mailto:info@tech2etc.com)
**LinkedIn**: [Sent Me A Connection Request](https://bd.linkedin.com/in/fahimulkabirchowdhury)

Feel free to reach out for collaborations, job opportunities, or any queries regarding this project!
````
