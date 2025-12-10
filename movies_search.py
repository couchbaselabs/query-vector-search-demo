import os
from datetime import timedelta
from typing import Any, Dict, Tuple

import streamlit as st
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from langchain_couchbase import CouchbaseQueryVectorStore
from langchain_couchbase.vectorstores import DistanceStrategy


def check_environment_variable(variable_name):
    """Check if environment variable is set"""
    if variable_name not in os.environ:
        st.error(
            f"{variable_name} environment variable is not set. Please add it to the secrets.toml file"  # noqa: E501
        )
        st.stop()


def generate_embeddings(client, input_data):
    """Generate OpenAI embeddings for the input data"""
    response = client.embeddings.create(input=input_data, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def cleanup_poster_url(poster_url):
    """Convert from https://m.media-amazon.com/images/M/MV5BMDFkYTc0MGEtZmNhMC00ZDIzLWFmNTEtODM1ZmRlYWMwMWFmXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_UX67_CR0,0,67,98_AL_.jpg to https://m.media-amazon.com/images/M/MV5BMDFkYTc0MGEtZmNhMC00ZDIzLWFmNTEtODM1ZmRlYWMwMWFmXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_.jpg"""  # noqa: E501

    prefix = poster_url.split("_V1_")[0]
    suffix = poster_url.split("_AL_")[1]

    return prefix + suffix


@st.cache_resource(show_spinner="Connecting to Couchbase")
def connect_to_couchbase(connection_string, db_username, db_password):
    """Connect to couchbase"""

    auth = PasswordAuthenticator(db_username, db_password)
    options = ClusterOptions(auth)
    connect_string = connection_string
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster


@st.cache_resource(show_spinner="Connecting to Vector Store")
def get_couchbase_vector_store(
    _cluster,
    db_bucket,
    db_scope,
    db_collection,
    _embedding,
    distance_metric,
    text_key,
    embedding_key,
) -> CouchbaseQueryVectorStore:
    """Return the Couchbase Query vector store"""
    if not distance_metric:
        distance_metric = DistanceStrategy.COSINE
    vector_store = CouchbaseQueryVectorStore(
        cluster=_cluster,
        bucket_name=db_bucket,
        scope_name=db_scope,
        collection_name=db_collection,
        embedding=_embedding,
        distance_metric=distance_metric,
        text_key=text_key,
        embedding_key=embedding_key,
    )
    return vector_store


@st.cache_resource
def create_filter(year_range: Tuple[int], rating: float) -> Dict[str, Any]:
    """Create a where clause for the hybrid search"""
    # Fields in the document used for search
    year_field = "Released_Year"
    rating_field = "IMDB_Rating"

    where_str = ""
    filter_operations = []
    if year_range:
        filter_operations.append(
            f"{year_field} >= {year_range[0]} AND {year_field} <= {year_range[1]}"
        )

    if rating:
        filter_operations.append(f"{rating_field} >= {rating}")

    if filter_operations and len(filter_operations) > 1:
        where_str = " AND ".join(filter_operations)
    elif filter_operations and len(filter_operations) == 1:
        where_str = filter_operations[0]

    return where_str


if __name__ == "__main__":
    st.set_page_config(
        page_title="Movie Search",
        page_icon="🎥",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    # Load environment variables
    DB_CONN_STR = os.getenv("DB_CONN_STR")
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_BUCKET = os.getenv("DB_BUCKET")
    DB_SCOPE = os.getenv("DB_SCOPE")
    DB_COLLECTION = os.getenv("DB_COLLECTION")
    # INDEX_NAME = os.getenv("INDEX_NAME")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    # Use text-embedding-3-small as the embedding model if not set
    if not EMBEDDING_MODEL:
        EMBEDDING_MODEL = "text-embedding-3-small"

    # Ensure that all environment variables are set
    check_environment_variable("OPENAI_API_KEY")
    check_environment_variable("DB_CONN_STR")
    check_environment_variable("DB_USERNAME")
    check_environment_variable("DB_PASSWORD")
    check_environment_variable("DB_BUCKET")
    check_environment_variable("DB_SCOPE")
    check_environment_variable("DB_COLLECTION")

    # Initialize empty filters
    search_filters = {}

    # Native OpenAI library for generating embeddings
    openai_embedding_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    embedding = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
    )

    # Connect to Couchbase Vector Store
    cluster = connect_to_couchbase(DB_CONN_STR, DB_USERNAME, DB_PASSWORD)
    bucket = cluster.bucket(DB_BUCKET)
    scope = bucket.scope(DB_SCOPE)

    # UI Elements
    text = st.text_input("Find your movie")
    with st.sidebar:
        st.header("Search Options")
        distance_metric = st.selectbox(
            "Distance Metric",
            [
                DistanceStrategy.DOT,
                DistanceStrategy.EUCLIDEAN,
                DistanceStrategy.COSINE,
                DistanceStrategy.EUCLIDEAN_SQUARED,
            ],
        )
        no_of_results = st.number_input(
            "Number of results", min_value=1, value=5, format="%i"
        )
        # Filters
        st.subheader("Filters")
        enable_filters = st.checkbox("Enable filters")

        if enable_filters:
            year_range = st.slider("Released Year", 1900, 2024, (1900, 2024))
            rating = st.number_input("Minimum IMDB Rating", 0.0, 10.0, 0.0, step=1.0)
            show_filter = st.checkbox("Show filter")
            hybrid_search_filter = create_filter(year_range, rating)
            if show_filter:
                st.text(hybrid_search_filter)

    submit = st.button("Submit")

    if submit:
        # Create the LangChain Couchbase Vector Store object
        vector_store = get_couchbase_vector_store(
            cluster,
            DB_BUCKET,
            DB_SCOPE,
            DB_COLLECTION,
            embedding,
            distance_metric,
            text_key="Overview",
            embedding_key="Overview_embedding",
        )

        # Fetch the filters
        if enable_filters:
            search_filters = create_filter(year_range, rating)

        # Perform the search using LangChain
        docs = vector_store.similarity_search_with_score(
            text,
            k=no_of_results,
            where_str=search_filters,
            fields=[
                "Series_Title",
                "Poster_Link",
                "Overview",
                "Released_Year",
                "IMDB_Rating",
                "Runtime",
            ],
        )

        for doc in docs:
            movie, distance = doc

            # Display the results in a grid
            st.header(movie.metadata["Series_Title"])
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    cleanup_poster_url(movie.metadata["Poster_Link"]),
                    width='stretch',
                )
            with col2:
                st.write("Synopsis:", movie.page_content)
                st.write(f"Distance: {distance:.{3}f}")
                st.write("Released Year:", movie.metadata["Released_Year"])
                st.write("IMDB Rating:", movie.metadata["IMDB_Rating"])
                st.write("Runtime:", movie.metadata["Runtime"])
            st.divider()
