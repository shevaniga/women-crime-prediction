# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from markupsafe import Markup
from sklearn.metrics import classification_report, confusion_matrix

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for flash messages

# Load model and preprocessors
model = pickle.load(open("models/final_model.pkl", "rb"))
scaler = pickle.load(open("models/final_scaler.pkl", "rb"))
le = pickle.load(open('models/final_encoder.pkl', 'rb'))
encoder = pickle.load(open("models/final_encoder.pkl", "rb"))
important_features = pickle.load(open("models/final_important_features.pkl", "rb"))
feature_importances_df = pickle.load(open('models/final_feature_importances.pkl', 'rb'))

# Load dataset
df = pd.read_csv("womencrimes.csv")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Visualizations
def generate_visualizations():
    corr = df[numerical_cols].corr()
    corr.index = [col.replace('_', ' ').title() for col in corr.index]
    corr.columns = [col.replace('_', ' ').title() for col in corr.columns]
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap", color_continuous_scale="ice")
    heatmap_html = fig.to_html(full_html=False)
    feature_importances_df['Feature'] = feature_importances_df['Feature'].str.replace('_', ' ').str.title()

    # Feature Importances (only if model has it)
    if feature_importances_df is not None:

        fig_bar = px.bar(feature_importances_df, x="Importance", y="Feature", orientation="h", title="Importance", color="Importance", color_continuous_scale="ice")
        bar_html = fig_bar.to_html(full_html=False)
    else:
        bar_html = "<p>Feature importance not matching feature names.</p>"

    return heatmap_html, bar_html

# Model Evaluation
def model_evaluation():
    df_eval = pd.read_csv('womencrimes.csv')
    for col in ['Area_Name', 'Group_Name', 'Sub_Group_Name']:
        df_eval[col] = df_eval[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
        if 'Unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'Unknown')
        df_eval[col] = le.transform(df_eval[col])

    df_eval[numerical_cols] = scaler.transform(df_eval[numerical_cols])
    X = df_eval[important_features]
    y_pred = model.predict(X)

    cm = confusion_matrix(y_pred.round(), y_pred.round(), labels=[0, 1])
    cm_fig = ff.create_annotated_heatmap(z=cm, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"], colorscale='blues')
    cm_html = cm_fig.to_html(full_html=False)

    report = classification_report(y_pred.round(), y_pred.round(), output_dict=True)
    accuracy = report['accuracy']

    return round(accuracy * 100, 2), cm_html

# Dark Theme Setup
def set_dark_theme():
    import plotly.io as pio
    dark_template = pio.templates["plotly_dark"]
    dark_template.layout.paper_bgcolor = "#0a192f"  # Dark Navy background
    dark_template.layout.plot_bgcolor = "#0a192f"   # Dark Navy background
    dark_template.layout.font.color = "#ffffff"     # White text
    dark_template.layout.colorway = ["#64ffda"]     # Light blue accents
    pio.templates["custom_dark"] = dark_template
    pio.templates.default = "custom_dark"


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    set_dark_theme()
    heatmap_html, bar_html = generate_visualizations()
    accuracy, cm_html = model_evaluation()

    if request.method == "POST":
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            if file.filename.endswith(".csv"):
                try:
                    new_data = pd.read_csv(file)
                    new_data_encoded = new_data.copy()

                    # Ensure all expected columns exist
                    for col in categorical_cols:
                        if col not in new_data_encoded.columns:
                            new_data_encoded[col] = 'Unknown'

                    for col in numerical_cols:
                        if col not in new_data_encoded.columns:
                            new_data_encoded[col] = 0  # Or np.nan if you prefer

                    # Encode categorical columns
                    for col in categorical_cols:
                        if col in new_data_encoded.columns:
                            new_data_encoded[col] = new_data_encoded[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                            new_data_encoded[col] = le.transform(new_data_encoded[col])


                    # Scale numerical columns
                    new_data_encoded[numerical_cols] = scaler.transform(new_data_encoded[numerical_cols])

                    # Use only important features
                    X_new = new_data_encoded[important_features]
                    preds = model.predict(X_new)

                    # Save predictions
                    new_data["Prediction"] = preds
                    output_path = os.path.join("static", "predictions.csv")
                    new_data.to_csv(output_path, index=False)

                    flash("CSV predictions generated successfully!", "success")
                    return redirect(url_for("predictions"))

                except Exception as e:
                    flash(f"Error processing the uploaded CSV: {e}", "error")
                    return redirect(url_for("index"))
            else:
                flash("Only CSV files are allowed.", "error")
        else:
            try:
                input_data = []
                for col in numerical_cols + categorical_cols:
                    val = request.form.get(col)
                    if col in numerical_cols:
                        input_data.append(float(val))
                    else:
                        input_data.append(val)

                # Prepare manual input
                input_df = pd.DataFrame([input_data], columns=numerical_cols + categorical_cols)

                # Encode categorical manually
                for col in categorical_cols:
                    input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                    input_df[col] = le.transform(input_df[col])


                # Scale numericals manually
                input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

                # Use only important features
                X_input = input_df[important_features]
                prediction = model.predict(X_input)[0]

                flash("Manual prediction successful!", "success")

            except Exception as e:
                flash(f"Error in manual prediction: {e}", "error")

    # Display dataset
    df_pretty = df.copy()
    df_pretty.columns = [col.replace('_', ' ').title() for col in df_pretty.columns]
    df_head = df_pretty.head().to_html(
        classes="table w-full text-sm text-left text-gray-700 border-separate border-spacing-y-2",
        border=0,
        index=False,
        escape=False
    )

    return render_template("index.html",
                            df_head=df_head,
                            prediction=prediction,
                            numerical=numerical_cols,
                            categorical=categorical_cols,
                            heatmap_html=Markup(heatmap_html),
                            bar_html=Markup(bar_html),
                            accuracy=accuracy,
                            cm_html=Markup(cm_html))


@app.route("/predictions")
def predictions():
    return render_template("predictions.html")

@app.route("/more")
def more():
    set_dark_theme()

    cleaned_df = df.copy()
    cleaned_df.columns = [col.replace('_', ' ').title() for col in cleaned_df.columns]

    # Histogram
    hist_fig = px.histogram(cleaned_df, x='Persons Convicted', title='Distribution of Persons Convicted')
    hist_html = hist_fig.to_html(full_html=False)

    # Box Plot
    box_fig = px.box(cleaned_df, x='Group Name', y='Persons Convicted', title='Convictions by Crime Group')
    box_html = box_fig.to_html(full_html=False)

    # Violin Plot
    violin_fig = px.violin(cleaned_df, x='Sub Group Name', y='Persons Convicted',
                           box=True, points="all", title='Violin Plot of Convictions by Sub Group')
    violin_html = violin_fig.to_html(full_html=False)

    # Scatter Plot
    scatter_fig = px.scatter(cleaned_df, x='Persons Arrested', y='Persons Convicted',
                             color='Group Name',
                             title='Arrests vs Convictions Scatter Plot')
    scatter_html = scatter_fig.to_html(full_html=False)

    # Bar – Average per Area
    avg_df = cleaned_df.groupby("Area Name")["Persons Convicted"].mean().reset_index()
    bar_avg_fig = px.bar(avg_df, x="Persons Convicted", y="Area Name", orientation="h",
                         title="Average Convictions per Area", color="Persons Convicted",
                         color_continuous_scale="ice")
    bar_avg_html = bar_avg_fig.to_html(full_html=False)

    # KDE Plot
    import seaborn as sns
    import matplotlib.pyplot as plt
    import io
    import base64

    fig_kde, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data=cleaned_df, x='Persons Convicted', fill=True, color="#64ffda", ax=ax)
    ax.set_title('KDE Plot - Conviction Density')
    ax.set_facecolor('#0a192f')
    fig_kde.patch.set_facecolor('#0a192f')
    ax.title.set_color('#64ffda')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    kde_buf = io.BytesIO()
    plt.savefig(kde_buf, format="png", bbox_inches="tight", facecolor=fig_kde.get_facecolor())
    kde_buf.seek(0)
    kde_html = f'<img src="data:image/png;base64,{base64.b64encode(kde_buf.read()).decode()}" class="rounded shadow-lg w-full"/>'
    plt.close(fig_kde)

    # 3D Scatter Plot
    scatter3d_fig = px.scatter_3d(cleaned_df,
                                  x='Persons Arrested',
                                  y='Persons Chargesheeted',
                                  z='Persons Convicted',
                                  color='Group Name',
                                  title='3D Plot: Arrested vs Chargesheeted vs Convicted')
    scatter3d_html = scatter3d_fig.to_html(full_html=False)

    # Pairplot (image via seaborn)
    pairplot_fig = sns.pairplot(cleaned_df[['Persons Arrested', 'Persons Chargesheeted', 'Persons Convicted', 'Group Name']], hue='Group Name', palette='coolwarm')
    pairplot_buf = io.BytesIO()
    pairplot_fig.savefig(pairplot_buf, format='png', bbox_inches='tight')
    pairplot_buf.seek(0)
    pairplot_html = f'<img src="data:image/png;base64,{base64.b64encode(pairplot_buf.read()).decode()}" class="rounded shadow-lg w-full"/>'
    plt.close()

    return render_template("more.html",
                           hist_html=Markup(hist_html),
                           box_html=Markup(box_html),
                           violin_html=Markup(violin_html),
                           scatter_html=Markup(scatter_html),
                           bar_avg_html=Markup(bar_avg_html),
                           kde_html=Markup(kde_html),
                           scatter3d_html=Markup(scatter3d_html),
                           pairplot_html=Markup(pairplot_html))




if __name__ == "__main__":
    app.run(debug=True)



