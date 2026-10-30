import plotly.express as px


def create_cluster_plot(
    df,
    labels,
    x_column,
    y_column,
    title
):

    plot_df = df.copy()

    plot_df["Cluster"] = labels.astype(str)

    fig = px.scatter(
        plot_df,
        x=x_column,
        y=y_column,
        color="Cluster",
        hover_data=[
            "User_ID",
            "Name",
            "Subscription_Type",
            "Favorite_Genre",
            "Watch_Time_Hours"
        ],
        title=title
    )

    return fig


def create_3d_plot(
    df,
    labels
):

    plot_df = df.copy()

    plot_df["Cluster"] = labels.astype(str)

    fig = px.scatter_3d(
        plot_df,
        x="Age",
        y="Watch_Time_Hours",
        z="Genre_Encoded",
        color="Cluster",
        hover_data=[
            "User_ID",
            "Name",
            "Subscription_Type",
            "Favorite_Genre"
        ],
        title="3D Customer Cluster Visualization"
    )

    return fig