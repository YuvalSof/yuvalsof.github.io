---
layout: post
title: "Range Against the Machine Case 2 - Beating AutoML by Looking Under the Hood"
date: 2025-10-03
categories: [case studies, modeling]
tags: [machine learning, business analytics, feature engineering,  Used Cars Dataset, Missing-Values Imputation]
author: yuval
---
<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/image 21.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>
---
**Used-car pricing prediction with business signals, not just buttons to click**

In this article, we'll explore several key data preprocessing techniques. We will cover the following topics:
* Hierarchical Missing Value Imputation 
* Manual Missing Value Imputation
* Automated Missing Value Imputation 
* Business Domain-Driven Feature Engineering
* Generative AI-Based Data Enrichment

**Will it be enough to beat the machine again?**
---
Again, I was officially assigned by the Massachusetts Institute of Technology to re-examine an iconic dataset, this time the India used-cars dataset, widely mirrored and originally scraped from CarDekho listings. In practice, the most common variant is a 7.2k × 14 table (Name, Location, Year, Kilometers_Driven, Fuel_Type, Transmission, Owner, Mileage, Engine, Power, Seats, New_Price, Price -  target) that has circulated in tutorials since ~2019–2020. Licensing varies by Kaggle mirror.
Performance-wise, tree-based regressors routinely reach **~0.93–0.94 R²**, whereas straightforward linear models can drop **below 0.70 R²**. The name of the game here is rigorous cleaning and missing-value imputation. 
In this article, the added value you won't find in common cover versions, is a little bit of used-car domain sense, the kind anyone who's bought a second-hand car once or twice (or five times, in my case) brings to the table.

## The Car that Circled India 285 Times at 247 km/h or Why We Shouldn't Skip df.describe().T

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/image 22.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

Before the deep dive, I like to eyeball the data. A quick describe().T served up a spit-take: five zeros in a row. According to the dataset, a 2017 BMW X5 had racked up **6.5 million km** and was listed for ₹65 lakh. To do that by 2020, it would have needed to average ~247 km/h nonstop-no fuel stops, no maintenance, no sleep, no bathroom breaks. Even with relay drivers, that's not "unlikely," it's impossible.
So…someone fat-fingered a zero. Or two. (If three - send me the seller's number; that's a steal.) Long story short, I treated it as an outlier and replaced it with the mean kilometers for 2017 models.

## Leveraging Messy 'Name' String for Missing-Value Imputation - Parsing Chaos into Signal

A brief glance at the dataset reveals that we have 2 main challenges in the dataset:
* The field 'Name' hides valuable information like make and model, but contains 2041 unique values. That is, due to inconsistent entries. Some would specify sub-models in the listing and some wouldn't. Even if we narrow it to make and model, it would result in too many categories for our dataset.

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/image 23.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

* We have too many missing values, especially in the 'New_Price' variable, that showed the highest correlation with our target - Price

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/image 24.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

**How much signal hides in "Name"? More than you'd think**

### Exact sub-model match.
 A bunch of the fields with gaps are basically factory specs - New_price, Mileage, Engine, Power, Seats. Within the same sub-model, these barely change. So if two listings share the exact 'Name' (e.g., BMW X5 xDrive30d M Sport), we can borrow specs from the ones that have them. First pass is dead simple: impute by the exact sub-model-use the mean for numeric fields and the mode for categoricals.
It's low-tech but high-yield: squeezing clean, reliable values out of the messiest column in the dataset.

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/code 1.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>
---

## Hierarchical Missing-Value Imputation

'Name' is all over the place. The same car shows up as **"Toyota Corolla Altis 1.8 G CNG"** or just **"Toyota Corolla Altis."** Sometimes one trim is missing a value while a sibling trim has everything. So instead of guessing, we start narrow and widen the circle: fill from the closest sub-model first, then back off step by step if that exact variant isn't available.

**Step 1: Build simple prefixes from 'Name'**- the first 4, 3, and 2 tokens-and impute in that order (numeric → median, categorical → mode). Example for "Toyota Corolla Altis 1.8 G CNG":
* **Model_4:** "Toyota Corolla Altis 1.8"
* **Model_3:** "Toyota Corolla Altis"
* **Model_2:** "Toyota Corolla"

The rule is **closest match wins:** try **Model_4;** if still missing, fall back to **Model_3;** then **Model_2;** only then consider broader groups (Make) or a global fallback. It's lightweight, auditable, and surprisingly powerful at turning messy names into dependable fills.

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/code 2.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

**Step 2: Fill the gaps with closest-match first.**
 Now we use the hierarchy we built to impute the blanks. For each missing value, I search inside the most specific bucket first:

**Model_4 → Model_3 → Model_2 → (optional) Make → global fallback.**

I used mode in all cases, but you can try other methods: Numerics get the median, categoricals the mode. If a bucket is too small (or empty), I back off to the next level. This "closest match wins" approach preserves detail when it exists and stays sane when it doesn't.

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/code 3.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

Our missing values balance, before and after:

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/image 25.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

## Leveraging "Name" + Gen-AI for Data Enrichment

Next, I used the raw 'Name' to tag each car with a class (Sub/Mini-compact, Compact, Minivan, Mid-Size, Large, Small SUV, Standard SUV, Sport/Two-Seater). Different classes target different buyers and often depreciate differently, so this feature can add lift for used-price modeling. Rather than pick one official taxonomy (Euro NCAP, US EPA, China, etc.), I built a compact, balanced schema that keeps categories frequent enough to be useful.
How it works (lightweight and reproducible):

* Noise filter / normalization: Strip engine badges, trims, drivetrains, years, and other junk (e.g., VTEC, CRDi, AT, 4x4) to isolate the core make–model - same idea as the hierarchical imputation step.
* Dictionary + back-off: Use an LLM-assisted dictionary to map cleaned names to classes, then apply the same hierarchical back-off (most specific → broader) to fill what the exact key misses.

This captured about five-sixths of rows, leaving ~1/6 unknown for later handling. I won't paste the long LLM-generated script here, but it's in my GitHub. Moral of the story: a pinch of domain sense plus Gen-AI scaffolding = fast, low-effort enrichment. The gain was modest - but it cost me almost nothing to implement.

### Other Missing-Value Imputation

I still had 12 missing Power values. Since 'Engine' and 'Power' move together (see the heatmap), I used engine size as a proxy. First, I looked for an exact displacement match (e.g., 1798 cc). If none existed, I widened the net to ±70 cc and took the median Power from those near-neighbors.
Why this works: the ±70 cc window is tight enough to keep comparable trims, but wide enough to find comps; using the median keeps outliers from skewing the fill.

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/code 4.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

The next hurdle was imputing the **1,512 missing New_price** values. A quick heatmap shows New_price is strongly tied to Power, but using a single feature gives a partial picture.
Brands carry big, stable price premia, so I first try to match Make + Power (with a tolerance), and only then fall back to power-only matches. The "business" logic is obvious: a ~150 hp **Tata Hexa XT** is not priced like a ~150 hp **Land Rover Discovery Sport TD4 HSE 7S** when new - brand matters.
You can see the brand stratification in the tails: among cars >300 hp, the counts skew to German luxury makes.

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/image 26.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

below **100 hp**, we see **Maruti, Hyundai, Honda** dominate - affordable mass-market brands. That means the high overall correlation between Power and New_price is **heterogeneous across brands**. In practice, I use a **hierarchical imputation:** (1) fill by median within the same Make and similar Power (±hp window), then (2) fall back to power-only matches if needed, and finally (3) a global fallback. A narrow hp window leans more on power similarity (and may cross brands); a wider window increases the chance of finding same-brand matches, effectively giving more weight to Make even if the hp gap is larger.

Step 1: Create 'Make' Variable:

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/code 5.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

Cleaning the variable:

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/code 6.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

Step 2: Using 'Make' in Missing-Value Imputation

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/code 7.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

## Feature Engineering

**Mileage matters - just not equally for everyone**

Mileage is a parameter for fuel efficiency. High mileage means that the car uses less fuel to drive more kilometers. 
A base model I preliminary ran on the dataset showed a very limited effect of Mileage on the target - Price. We know that fuel efficiency is a very desired feature in cars. When I look for a new car, I always check its efficiency and it plays a crucial role in my decision making. Most  of us, when we buy a car, ask Immediately "how much fuel does it drink". Who doesn't? 
Well… some don't. People who are willing to spend enormous amounts on luxury cars, apparently have different priorities. 
Therefore I decided to create an interaction between luxurious makes and Mileage:

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/code 8.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

**Age**

For explanatory reasons only I replaced Year with Age.

**KM Driven Per Year**

Buyers prefer less worn-out cars - cars that sat in the garage most of the time, and were driven only around the neighborhood, and to annual checkups. I know it, because this is what I ask for when I go to buy a car at a car dealership. The wear-out is relative to the number age of the car.

**KM Driven Per Year - Is Luxury Interaction**

Let's see if luxury cars have different detraction pattern when it comes to relatively high km driven.

**Log Transform**

The target (Price) and features (KM_Driven, New_price) were heavily right-skewed, so I applied a log transform (natural log) to reduce skew and stabilize variance. This tends to linearize relationships and makes the model's fit and residuals behave better.

The log transforming improved the skewness of the variables:

**Before:**

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/image 27.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>


**After:**

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/image 28.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

A real makeover…

---

## Range Against the Machine

With preprocessing wrapped, the next step was to see how much lift it actually delivered. As before, I benchmarked against **MLJAR AutoML**. MLJAR advertises built-in preprocessing - missing-value imputation and feature engineering included - which is exactly where most of the effort went in this dataset.
Still, I gave it a fair shot by supplying a **generic, one-size-fits-all** pipeline suggested by my favorite LLM. The pipeline uses **Iterative Imputer (MICE)** for numerical features - modeling missing values from observed ones - while imputing categorical features with the mode. It also keeps missingness indicators to preserve signal and produces plausible estimates for nulls without leaking targets.
You can find the full code in my Github Repo, link below.

I ran AutoML twice: (1) on the cleaned dataset (only a log transform added), and (2) on the dataset processed by the generic MICE-style imputation pipeline.

## Feature Engineering Results

Before we reveal the final numbers, a brief pit stop: how much did the feature engineering actually help?
The SHAP plot below illustrates the contribution of the newly engineered features.

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/image 29.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

All and all, the engineered featured had a modest contribution to the model. 
The Mileage-Non luxury interaction gave the fuel efficiency factor more explanatory power and pushed it to №7, vs the previous №10. 
Our AI-aided Class feature added some signal, suggesting (but not proving) different depreciation patterns for different classes. 
Our KM Driven Per Year features added some signal on top of KM Driven, while surprisingly, we see that Luxury cars, with higher KM driven per year are a predictor of higher selling price. How come?
This is the opposite of the typical expectation for used car price (where higher mileage usually decreases value), which implies that Luxury cars are not more sensitive (in a negative value sense) to high KM per year, but rather, high KM per year provides a net positive or less negative signal in this model, possibly because:
* It indicates frequent maintenance or good running condition.
* It acts as a proxy for newer cars (since the feature is annual mileage, not total odometer reading, and newer cars are often driven more annually).

## Model Results
**AutoML (standalone)** delivered the weakest results: R² < 0.80 with the Decision Tree Regressor, and around **R² ≈ 0.85** for both the Random Forest Regressor and the linear model. In contrast, its **XGBoost** model performed excellent, achieving **R² > 0.95.**

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/image 291.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

**AutoML + MICE-style aid** improved every model. The lift was most pronounced for the **linear model,** which reached **R² > 0.90.**
**XGBoost (AutoML)** was best-in-class, delivering an excellent **R² > 0.966.**

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/image 292.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

These results are striking: a simple, generic, one-size-fits-all pipeline-generated via prompt with near-zero effort- **boosted AutoML's performance** despite its own built-in imputation.

**But was it enough?**

To my surprise, my boutique, hand-crafted models outperformed AutoML by a wide margin. Every model cleared **R² = 0.93** - including the linear ones. The decision tree was the weakest and least stable (why? see my article, If a Decision Tree Falls in a Random Forest), while my XGboost delivered an outperforming **R² > 0.97.**

<div style="width: 100%; max-width: 60%; margin: 0 auto; overflow: hidden; height: 90%; position: relative;">
  <img src="/assets/img/posts/range-against-the-machine2/image 293.png"
       alt="Alt text"
       style="width: 100%; position: relative; top: 0; height: 111.11%;">
</div>

## Summary & Conclusions

* The biggest predictive performance gains came from my hand-crafted, hierarchical missing-value imputation. It didn't require complex Python - just common sense and a careful look at the features.

* Despite its simplicity, it outperformed robust ML-aided approaches such as MICE and MLJAR's built-in preprocessing.

* A dose of domain reasoning (the kind a seasoned used-car buyer applies), combined with AI-assisted feature enrichment, added explanatory power and pushed accuracy even further.

---

---

Range Against the Machine Case 2: Beating AutoML by Looking Under the Hood
Used-car pricing prediction with business signals, not just buttons to click
In this article, we'll explore several key data preprocessing techniques. We will cover the following topics:
Hierarchical Missing Value Imputation 
Manual Missing Value Imputation
Automated Missing Value Imputation 
Business Domain-Driven Feature Engineering
Generative AI-Based Data Enrichment

Will it be enough to beat the machine again?

---

Again, I was officially assigned by the Massachusetts Institute of Technology to re-examine an iconic dataset, this time the India used-cars dataset, widely mirrored and originally scraped from CarDekho listings. In practice, the most common variant is a 7.2k × 14 table (Name, Location, Year, Kilometers_Driven, Fuel_Type, Transmission, Owner, Mileage, Engine, Power, Seats, New_Price, Price -  target) that has circulated in tutorials since ~2019–2020. Licensing varies by Kaggle mirror.
Performance-wise, tree-based regressors routinely reach ~0.93–0.94 R², whereas straightforward linear models can drop below 0.70 R². The name of the game here is rigorous cleaning and missing-value imputation. 
In this article, the added value you won't find in common cover versions, is a little bit of used-car domain sense, the kind anyone who's bought a second-hand car once or twice (or five times, in my case) brings to the table.
The Car that Circled India 285 Times at 247 km/h or Why We Shouldn't Skip df.describe().T
Before the deep dive, I like to eyeball the data. A quick describe().T served up a spit-take: five zeros in a row. According to the dataset, a 2017 BMW X5 had racked up 6.5 million km and was listed for ₹65 lakh. To do that by 2020, it would have needed to average ~247 km/h nonstop-no fuel stops, no maintenance, no sleep, no bathroom breaks. Even with relay drivers, that's not "unlikely," it's impossible.
So…someone fat-fingered a zero. Or two. (If three - send me the seller's number; that's a steal.) Long story short, I treated it as an outlier and replaced it with the mean kilometers for 2017 models:
# Replacing the 6.5M Kilometers_Driven observation with mean in cohort

df.loc[df['Kilometers_Driven'] == 6500000, 'Kilometers_Driven'] = (
    df[df['Year'] == 2017]['Kilometers_Driven'].mean()
)
Leveraging Messy 'Name' String for Missing-Value Imputation - Parsing Chaos into Signal
A brief glance at the dataset reveals that we have 2 main challenges in the dataset:
The field 'Name' hides valuable information like make and model, but contains 2041 unique values. That is, due to inconsistent entries. Some would specify sub-models in the listing and some wouldn't. Even if we narrow it to make and model, it would result in too many categories for our dataset.

We have too many missing values, especially in the 'New_Price' variable, that showed the highest correlation with our target - Price

How much signal hides in "Name"? More than you'd think.
Exact sub-model match.
 A bunch of the fields with gaps are basically factory specs - New_price, Mileage, Engine, Power, Seats. Within the same sub-model, these barely change. So if two listings share the exact 'Name' (e.g., BMW X5 xDrive30d M Sport), we can borrow specs from the ones that have them. First pass is dead simple: impute by the exact sub-model-use the mean for numeric fields and the mode for categoricals.
It's low-tech but high-yield: squeezing clean, reliable values out of the messiest column in the dataset.
# Replace Mileage Nans with mode in observation with the same Name
df['Mileage'] = (
    df.groupby('Name')['Mileage']
      .transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
)

# Same for Engine
df['Engine'] = (
    df.groupby('Name')['Engine']
      .transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
)

# Same for Power
df['Power'] = (
    df.groupby('Name')['Power']
      .transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
)

# Same for Seats
df['Seats'] = (
    df.groupby('Name')['Seats']
      .transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
)

# Same for New Price
df['New_price'] = (
    df.groupby('Name')['New_price']
      .transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
)

---

Hierarchical Missing-Value Imputation
'Name' is all over the place. The same car shows up as "Toyota Corolla Altis 1.8 G CNG" or just "Toyota Corolla Altis." Sometimes one trim is missing a value while a sibling trim has everything. So instead of guessing, we start narrow and widen the circle: fill from the closest sub-model first, then back off step by step if that exact variant isn't available.
Step 1: Build simple prefixes from 'Name'- the first 4, 3, and 2 tokens-and impute in that order (numeric → median, categorical → mode). Example for "Toyota Corolla Altis 1.8 G CNG":
Model_4: "Toyota Corolla Altis 1.8"
Model_3: "Toyota Corolla Altis"
Model_2: "Toyota Corolla"

The rule is closest match wins: try Model_4; if still missing, fall back to Model_3; then Model_2; only then consider broader groups (Make) or a global fallback. It's lightweight, auditable, and surprisingly powerful at turning messy names into dependable fills.
df['Model_2'] = df['Name'].apply(lambda x: ' '.join(x.split()[:2]))
df['Model_3'] = df['Name'].apply(lambda x: ' '.join(x.split()[:3]))
df['Model_4'] = df['Name'].apply(lambda x: ' '.join(x.split()[:4]))
Step 2: Fill the gaps with closest-match first.
 Now we use the hierarchy we built to impute the blanks. For each missing value, I search inside the most specific bucket first:
Model_4 → Model_3 → Model_2 → (optional) Make → global fallback.
I used mode in all cases, but you can try other methods: Numerics get the median, categoricals the mode. If a bucket is too small (or empty), I back off to the next level. This "closest match wins" approach preserves detail when it exists and stays sane when it doesn't.
# Imputes missing values in the Mileage using Model_4
df['Mileage'] = (
    df.groupby('Model_4')['Mileage']
      .transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
)

# Same for Engine
df['Engine'] = (
    df.groupby('Model_4')['Engine']
      .transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
)

# Same for Power
df['Power'] = (
    df.groupby('Model_4')['Power']
      .transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
)

# Same for Seats
df['Seats'] = (
    df.groupby('Model_4')['Seats']
      .transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
)

# Same for New Price
df['New_price'] = (
    df.groupby('Model_4')['New_price']
      .transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
)

# Repeat the same process with Model_3 and then Model_2
Our missing values balance, before and after:
Leveraging "Name" + Gen-AI for Data Enrichment
Next, I used the raw 'Name' to tag each car with a class (Sub/Mini-compact, Compact, Minivan, Mid-Size, Large, Small SUV, Standard SUV, Sport/Two-Seater). Different classes target different buyers and often depreciate differently, so this feature can add lift for used-price modeling. Rather than pick one official taxonomy (Euro NCAP, US EPA, China, etc.), I built a compact, balanced schema that keeps categories frequent enough to be useful.
How it works (lightweight and reproducible):
Noise filter / normalization: Strip engine badges, trims, drivetrains, years, and other junk (e.g., VTEC, CRDi, AT, 4x4) to isolate the core make–model - same idea as the hierarchical imputation step.
Dictionary + back-off: Use an LLM-assisted dictionary to map cleaned names to classes, then apply the same hierarchical back-off (most specific → broader) to fill what the exact key misses.

This captured about five-sixths of rows, leaving ~1/6 unknown for later handling. I won't paste the long LLM-generated script here, but it's in my GitHub. Moral of the story: a pinch of domain sense plus Gen-AI scaffolding = fast, low-effort enrichment. The gain was modest - but it cost me almost nothing to implement.
Other Missing-Value Imputation
I still had 12 missing Power values. Since 'Engine' and 'Power' move together (see the heatmap), I used engine size as a proxy. First, I looked for an exact displacement match (e.g., 1798 cc). If none existed, I widened the net to ±70 cc and took the median Power from those near-neighbors.
Why this works: the ±70 cc window is tight enough to keep comparable trims, but wide enough to find comps; using the median keeps outliers from skewing the fill.
# First pass: exact match on Engine
df['Power'] = 
df.groupby('Engine')['Power'].transform(lambda x: x.fillna(x.median()))

# Second pass: fill by Engine ± 70
def fill_by_engine_range(row):
    if pd.isna(row['Power']) and not pd.isna(row['Engine']):
        matches = df[(df['Engine'] >= row['Engine'] - 70) &
                     (df['Engine'] <= row['Engine'] + 70) &
                     df['Power'].notna()]
        if not matches.empty:
            return matches['Power'].median()
    return row['Power']

df['Power'] = df.apply(fill_by_engine_range, axis=1)
The next hurdle was imputing the 1,512 missing New_price values. A quick heatmap shows New_price is strongly tied to Power, but using a single feature gives a partial picture.
Brands carry big, stable price premia, so I first try to match Make + Power (with a tolerance), and only then fall back to power-only matches. The "business" logic is obvious: a ~150 hp Tata Hexa XT is not priced like a ~150 hp Land Rover Discovery Sport TD4 HSE 7S when new - brand matters.
You can see the brand stratification in the tails: among cars >300 hp, the counts skew to German luxury makes.
below 100 hp, we see Maruti, Hyundai, Honda dominate - affordable mass-market brands. That means the high overall correlation between Power and New_price is heterogeneous across brands. In practice, I use a hierarchical imputation: (1) fill by median within the same Make and similar Power (±hp window), then (2) fall back to power-only matches if needed, and finally (3) a global fallback. A narrow hp window leans more on power similarity (and may cross brands); a wider window increases the chance of finding same-brand matches, effectively giving more weight to Make even if the hp gap is larger.
Step 1: Create 'Make' Variable:
# Create a varible that extracts from Name all characters before space
 df['Make'] = df['Name'].str.split(' ').str[0]

 # All unique values of Make
 df['Make'].unique()
Cleaning the variable:
# Change values in Make: OpelCorsa = Opel, Force = Force One ....
df['Make'] = (
    df['Make'].replace(
        ['OpelCorsa', 'Force', 'Hindustan', 'ISUZU'],
        ['Opel', 'Force One', 'Hindustan Motors', 'Isuzu']
    )
)
Step 2: Using 'Make' in Missing-Value Imputation
#  Controls
PWR_TOL = 60  # +/- range for "nearby" power matches. 
# The higher the TOL, the more weight is on the make. 

# Exact Make+Power median 
df['New_price'] = (
    df.groupby(['Make', 'Power'])['New_price']
      .transform(lambda s: s.fillna(s.median()))
)

# Make + Power range (+/- PWR_TOL)
def fill_by_make_power_range(row):
    if pd.isna(row['New_price']) and pd.notna(row['Power']) and pd.notna(row['Make']):
        m = (
            (df['Make'] == row['Make']) &
            df['Power'].between(row['Power'] - PWR_TOL, row['Power'] + PWR_TOL) &
            df['New_price'].notna()
        )
        matches = df.loc[m, 'New_price']
        if not matches.empty:
            return matches.median()
    return row['New_price']

df['New_price'] = df.apply(fill_by_make_power_range, axis=1)

# Power-only exact median (fallback)
df['New_price'] = (
    df.groupby('Power')['New_price']
      .transform(lambda s: s.fillna(s.median()))
)

# Power-only range (+/- PWR_TOL) fallback
def fill_by_power_range(row):
    if pd.isna(row['New_price']) and pd.notna(row['Power']):
        m = (
            df['Power'].between(row['Power'] - PWR_TOL, row['Power'] + PWR_TOL) &
            df['New_price'].notna()
        )
        matches = df.loc[m, 'New_price']
        if not matches.empty:
            return matches.median()
    return row['New_price']

df['New_price'] = df.apply(fill_by_power_range, axis=1)

# Final global fallback  -----
df['New_price'] = df['New_price'].fillna(df['New_price'].median())

---

Feature Engineering
Mileage matters - just not equally for everyone
Mileage is a parameter for fuel efficiency. High mileage means that the car uses less fuel to drive more kilometers. 
A base model I preliminary ran on the dataset showed a very limited effect of Mileage on the target - Price. We know that fuel efficiency is a very desired feature in cars. When I look for a new car, I always check its efficiency and it plays a crucial role in my decision making. Most  of us, when we buy a car, ask Immediately "how much fuel does it drink". Who doesn't? 
Well… some don't. People who are willing to spend enormous amounts on luxury cars, apparently have different priorities. 
Therefore I decided to create an interaction between luxurious makes and Mileage:
# Define the luxury brands
Luxury = ['Audi', 'Land', 'Mercedes-Benz', 'BMW', 'Porsche',
          'Jaguar', 'Volvo', 'Mini', 'Jeep', 'Bentley', 'Lamborghini']

# Create binary column: 1 if Make is in Luxury list, else 0
dfn['Is_Luxury'] = dfn['Make'].isin(Luxury).astype(int)

dfn['Is_NonLuxury'] = 1 - dfn['Is_Luxury']

dfn['Mileage_NonLuxury_Interaction'] = dfn['Mileage'] * dfn['Is_NonLuxury']
Age
For explanatory reasons only I replaced Year with Age:
# Let's create a new variable to replace Year with Age - sbtract Year from current Year
from datetime import datetime
df['Age'] = datetime.now().year - df['Year']
 KM Driven Per Year
Buyers prefer less worn-out cars - cars that sat in the garage most of the time, and were driven only around the neighborhood, and to annual checkups. I know it, because this is what I ask for when I go to buy a car at a car dealership. The wear-out is relative to the number age of the car:
# Create a KM_Driven_Per_Year
df['KM_Driven_Per_Year'] = df['Kilometers_Driven'] / (2021 - df['Year'])
KM Driven Per Year - Is Luxury Interaction
Let's see if luxury cars have different detraction pattern when it comes to relatively high km driven.
# Create a an interaction between  KM_Driven_Per_Year and Is_Luxury
dfn['KM_Driven_Per_Year_Luxury_Interaction']  = dfn['KM_Driven_Per_Year'] * dfn['Is_Luxury']
Log Transform
The target (Price) and features (KM_Driven, New_price) were heavily right-skewed, so I applied a log transform (natural log) to reduce skew and stabilize variance. This tends to linearize relationships and makes the model's fit and residuals behave better.
# Log Price
df['Price_log'] = np.log1p(df['Price'])

# Same for New Price
df['New_price_log'] = np.log1p(df['New_price'])

# Sake for Kilometers Driven
df['Kilometers_Driven_log'] = np.log1p(df['Kilometers_Driven'])
The log transforming improved the skewness of the variables:
Before:
After:
A real makeover…

---

Range Against the Machine
With preprocessing wrapped, the next step was to see how much lift it actually delivered. As before, I benchmarked against MLJAR AutoML. MLJAR advertises built-in preprocessing - missing-value imputation and feature engineering included - which is exactly where most of the effort went in this dataset.
Still, I gave it a fair shot by supplying a generic, one-size-fits-all pipeline suggested by my favorite LLM. The pipeline uses Iterative Imputer (MICE) for numerical features - modeling missing values from observed ones - while imputing categorical features with the mode. It also keeps missingness indicators to preserve signal and produces plausible estimates for nulls without leaking targets.
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

def auto_fill_nulls(df: pd.DataFrame,
                    target: str = "Price",
                    drop_thresh: float = 0.80,
                    add_indicators: bool = True,
                    max_iter: int = 10,
                    random_state: int = 0):
    df = df.copy()

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame.")

    y = df[target]
    X = df.drop(columns=[target])

    # 1) drop super-sparse feature cols
    sparse_cols = [c for c in X.columns if X[c].isna().mean() > drop_thresh]
    if sparse_cols:
        X = X.drop(columns=sparse_cols)

    # 2) add missingness indicators (before filling)
    if add_indicators:
        for c in X.columns:
            if X[c].isna().any():
                X[f"{c}__was_na"] = X[c].isna().astype("int8")

    # 3) split numeric/categorical
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # 4) numeric imputation (MICE-like)
    num_imputer = IterativeImputer(
        random_state=random_state,
        initial_strategy="median",
        max_iter=max_iter
    )
    if num_cols:
        X_num = pd.DataFrame(
            num_imputer.fit_transform(X[num_cols]),
            columns=num_cols, index=X.index
        )
    else:
        X_num = pd.DataFrame(index=X.index)

    # 5) categorical imputation (mode)
    X_cat = X[cat_cols].copy()
    for c in cat_cols:
        mode = X_cat[c].mode(dropna=True)
        fill_val = mode.iloc[0] if not mode.empty else "Unknown"
        X_cat[c] = X_cat[c].fillna(fill_val)

    # 6) reassemble in original (X) order
    X_filled = pd.concat([X_num, X_cat], axis=1)[X.columns]

    # 7) cast types safely
    # a) indicators back to Int8
    for c in [col for col in X_filled.columns if col.endswith("__was_na")]:
        X_filled[c] = X_filled[c].round().astype("Int8")

    # b) original integer-like numeric cols: check against X (not df) to avoid KeyError
    for c in [col for col in num_cols if not col.endswith("__was_na")]:
        # restore only if the original column in X was integer dtype
        if c in X.columns and pd.api.types.is_integer_dtype(X[c].dropna()):
            X_filled[c] = np.round(X_filled[c]).astype("Int64")

    out = pd.concat([X_filled, y], axis=1)

    artifacts = {
        "dropped_columns": sparse_cols,
        "numeric_imputer": num_imputer
    }
    return out, artifacts
I ran AutoML twice: (1) on the cleaned dataset (only a log transform added), and (2) on the dataset processed by the generic MICE-style imputation pipeline.
Feature Engineering Results
Before we reveal the final numbers, a brief pit stop: how much did the feature engineering actually help?
The SHAP plot below illustrates the contribution of the newly engineered features.
All and all, the engineered featured had a modest contribution to the model. 
The Mileage-Non luxury interaction gave the fuel efficiency factor more explanatory power and pushed it to №7, vs the previous №10. 
Our AI-aided Class feature added some signal, suggesting (but not proving) different depreciation patterns for different classes. 
Our KM Driven Per Year features added some signal on top of KM Driven, while surprisingly, we see that Luxury cars, with higher KM driven per year are a predictor of higher selling price. How come?
This is the opposite of the typical expectation for used car price (where higher mileage usually decreases value), which implies that Luxury cars are not more sensitive (in a negative value sense) to high KM per year, but rather, high KM per year provides a net positive or less negative signal in this model, possibly because:
It indicates frequent maintenance or good running condition.
It acts as a proxy for newer cars (since the feature is annual mileage, not total odometer reading, and newer cars are often driven more annually).

Model Results
AutoML (standalone) delivered the weakest results: R² < 0.80 with the Decision Tree Regressor, and around R² ≈ 0.85 for both the Random Forest Regressor and the linear model. In contrast, its XGBoost model performed excellent, achieving R² > 0.95.

---

AutoML + MICE-style aid improved every model. The lift was most pronounced for the linear model, which reached R² > 0.90.
 XGBoost (AutoML) was best-in-class, delivering an excellent R² > 0.966.
These results are striking: a simple, generic, one-size-fits-all pipeline-generated via prompt with near-zero effort- boosted AutoML's performance despite its own built-in imputation.
But was it enough?

---

To my surprise, my boutique, hand-crafted models outperformed AutoML by a wide margin. Every model cleared R² = 0.93 - including the linear ones. The decision tree was the weakest and least stable (why? see my article, [If a Decision Tree Falls in a Random Forest](https://medium.com/python-in-plain-english/if-a-decision-tree-falls-in-a-random-forest-and-no-one-is-around-to-hear-did-it-make-a-sound-a282e5ab2e70){:target="_blank" rel="noopener"}), while my XGboost delivered an outperforming R² > 0.97.
Summary & Conclusions
The biggest predictive performance gains came from my hand-crafted, hierarchical missing-value imputation. It didn't require complex Python - just common sense and a careful look at the features.
Despite its simplicity, it outperformed robust ML-aided approaches such as MICE and MLJAR's built-in preprocessing.
A dose of domain reasoning (the kind a seasoned used-car buyer applies), combined with AI-assisted feature enrichment, added explanatory power and pushed accuracy even further.

---

**Data Dictionary**
**S.No.** : Serial Number
**Name :** Name of the car which includes Brand name and Model name
**Location :** The location in which the car is being sold or is available for purchase (Cities)
**Year :** Manufacturing year of the car
**Kilometers_driven :** The total kilometers driven in the car by the previous owner(s) in KM
**Fuel_Type :** The type of fuel used by the car (Petrol, Diesel, Electric, CNG, LPG)
**Transmission :** The type of transmission used by the car (Automatic / Manual)
**Owner :** Type of ownership
**Mileage :** The mileage offered by the car company in kmpl or km/kg
**Engine :** The displacement volume of the engine in CC
**Power :** The maximum power of the engine in bhp
**Seats :** The number of seats in the car
**New_Price :** The price of a new car of the same model in INR 100,000
**Price :** The price of the used car in INR 100,000 (Target Variable)

---

You can find the full code notebook and the dataset on my [GitHub Repo](https://github.com/YuvalSof/Used-Cars-Prediction){:target="_blank" rel="noopener"}