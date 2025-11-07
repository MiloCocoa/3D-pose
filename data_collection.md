# Barbell Squat Data Collection Plan
## 1. Goal
To create a high-quality, balanced dataset for training a mistake-detection model. **Variety and data quality are more important than raw quantity.**
## 2. The Prototype Dataset (Start Here)
Do not try to collect 1,000 repetitions at first. Our first goal is to build a small, high-quality "prototype" dataset to validate the entire pipeline.
- **Target:** ~10-15 repetitions for **each** of the 10 mistake classes.
- **Total Reps:** ~100-150 repetitions.
- **Subjects:** Must be from at least **2-3 different subjects**.
- **Purpose:** This small dataset is for testing. It lets us prove the model can learn (i.e., accuracy goes above 10%) before we spend weeks collecting more data.
## 3. The Full Dataset (Scaling Up)
Once the prototype proves the pipeline works, we can aim for a scale similar to the original paper (~30,000 frames).
- **Target:** ~100-150 repetitions for **each** of the 10 mistake classes.
- **Total Reps:** ~1,000 - 1,500 repetitions.
- **Subjects:** Aim for **10-15+ unique subjects** to ensure the model generalizes.
## 4. Key Principles for High-Quality Data
### Principle 1: Prioritize Variety
A varied dataset is the _only_ way to build a model that works in the real world.
- **Subject Variety:**
    - **Body Type:** Collect from people who are tall, short, heavy, and light.
    - **Experience Level:** Capture experienced lifters (who will have good "Correct" reps) and beginners (who will make "natural" mistakes).
    - **Gender:** Include a mix of genders to capture different body mechanics.
- **Environmental Variety (Crucial for Depth Cameras):**
    - **Camera Angle:** **Do not** film every squat from a perfect 90-degree side view. Capture data from:
        - Side (left and right)
        - Front-Quarter (45-degree angle)
        - Back-Quarter (45-degree angle)
    - **Clothing:** Ask subjects to wear form-fitting clothing (e.g., shorts, t-shirts). Baggy sweatpants or hoodies will "occlude" the joints and confuse the depth camera, leading to noisy data. Avoid all-black outfits if possible, as they can absorb IR light.
    - **Lighting:** Ensure the area is well-lit. Avoid strong backlighting (e.g., filming against a bright window).
### Principle 2: Ensure Class Balance (The Hardest Part)
You cannot have 1,000 "Correct" reps and only 10 "Butt Wink" reps. The model will just learn to always predict "Correct".
- **"Correct" Reps:** You need a large-and-diverse base of "perfect" (label 0) reps. These are your gold standard.
- **"Mistake" Reps:** You must _actively hunt_ for your 9 mistake classes.
    - **Observe Beginners:** They will provide many natural mistakes.
    - **Ask for Mistakes:** Don't be afraid to ask experienced lifters to _intentionally perform_ the mistakes (e.g., "On this set, can you let your knees cave in?"). This is a standard and effective way to get clean, isolated examples of each mistake class.
### Principle 3: Labeling is Part of Collection
The person filming _must_ be the one labeling.
- **Label Per Repetition:** After a subject does a set of 5 reps, the data collector must immediately watch the playback and label each rep (e.g., Rep 1: "Correct", Rep 2: "Correct", Rep 3: "Too Shallow", etc.).
- **One Mistake at a Time:** If a rep has _both_ "Butt Wink" and "Knees Inward," you must choose the _primary_ mistake. Don't try to multi-label (our model isn't set up for it).
## 5. Final Checklist
- [ ] Do I have a diverse set of subjects?
- [ ] Am I filming from multiple angles?
- [ ] Are subjects wearing form-fitting clothes?
- [ ] Do I have high-quality "Correct" reps from _every_ subject?
- [ ] Am I actively capturing _all 9_ mistake types?
- [ ] Are all classes (roughly) balanced in my dataset?
- [ ] Is every single repetition assigned a label?