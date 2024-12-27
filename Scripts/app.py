import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import linregress

merged_data = pd.read_csv('cleaned_merged_data.csv')

def regression_line(fig, x, y, df):
    slope, intercept, r_value, _, _ = linregress(df[x], df[y])
    line_x = [df[x].min(), df[x].max()]
    line_y = [slope * xi + intercept for xi in line_x]
    fig.add_trace(
        go.Scatter(x=line_x, y=line_y,
            mode="lines", name=f"Trend Line (r={r_value:.2f})",
            line=dict(color="red", dash="dash"),
        )
    )


def PHEvsLE_plotly(data=merged_data):
    fig = px.scatter(
        data,
        x='Health Expenditure (% GDP)', y='Life Expectancy',
        color='Country Name', symbol='Country Name',
        title='Public Health Expenditure vs Life Expectancy',
        labels={'Health Expenditure (% GDP)': 'Public Health Expenditure (% GDP)', 'Life Expectancy': 'Life Expectancy'},
        template='plotly_white',
    )
    regression_line(fig, 'Health Expenditure (% GDP)', 'Life Expectancy', data)
    fig.update_layout(height=600,width=900)
    return fig

def PHEvsH_plotly(data=merged_data):
    fig = px.scatter(
        data,
        x='Health Expenditure (% GDP)', y='Life Ladder',
        color='Country Name', symbol='Country Name',
        title='Public Health Expenditure vs Happiness',
        labels={'Health Expenditure (% GDP)': 'Public Health Expenditure (% GDP)', 'Life Ladder': 'Happiness'},
        template='plotly_white',
    )
    regression_line(fig, 'Health Expenditure (% GDP)', 'Life Ladder', data)
    
    fig.update_layout(height=600,width=900)
    return fig

def HvsLE_plotly(data=merged_data):
    fig = px.scatter(
        data,
        x='Life Expectancy', y='Life Ladder',
        color='Country Name', symbol='Country Name',
        title='Happiness vs Life Expectancy',
        labels={'Life Expectancy': 'Life Expectancy', 'Life Ladder': 'Happiness'},
        template='plotly_white',
    )
    regression_line(fig, 'Life Expectancy', 'Life Ladder', data)
    fig.update_layout(height=600,width=900)
    return fig

def OOPEvsCHE_plotly(data=merged_data):
    fig = px.scatter(
        data,
        x='OOP Expenditure', y='Current Health Expenditure',
        size='Health Expenditure (% GDP)', color='Country Name',
        title='OOP Expenditure vs Current Health Expenditure',
        labels={'OOP Expenditure': 'OOP Expenditure ($ PPP adjusted)', 'Current Health Expenditure': 'Current Health Expenditure'},
        template='plotly_white',
    )
    fig.update_layout(height=600,width=900)
    return fig  # not adding regression line since it won't make sense for this plot.

def HvsLEvsHE_plotly(data=merged_data):
    fig = px.scatter(
        data,
        x='Health Expenditure (% GDP)', y='Life Expectancy',
        size='Life Ladder', color='Country Name',
        title='Life Expectancy vs Health Expenditure (% GDP) and Happiness',
        labels={'Health Expenditure (% GDP)': 'Public Health Expenditure (% GDP)', 'Life Expectancy': 'Life Expectancy'},
        template='plotly_white',
    )
    regression_line(fig, 'Health Expenditure (% GDP)', 'Life Expectancy', data)
    fig.update_layout(height=600,width=900)
    return fig

def over_time_line_chart(data):
    y_axis_metric = st.selectbox(
        "Select Metric for Y-Axis",
        options=[
            'Life Expectancy', 'Life Ladder', 'Health Expenditure (% GDP)', 'OOP Expenditure', 'Current Health Expenditure'
        ],
        index=0
    )

    fig = px.line(
        data, x='Year', y=y_axis_metric,
        color='Country Name', title=f'{y_axis_metric} Over Time', labels={'Year': 'Year', y_axis_metric: y_axis_metric},
        template='plotly_white'
    )
    fig.update_traces(mode='lines+markers')
    fig.update_layout(height=600, width=900)
    st.plotly_chart(fig, use_container_width=True)

def top10_public_health_spending(data=merged_data):
    spending_data = data.groupby(['Country Name'])['Health Expenditure (% GDP)'].mean().reset_index()
    spending_data = spending_data.sort_values(by='Health Expenditure (% GDP)', ascending=False)
    fig = px.bar(
        spending_data.head(10),
        x='Country Name', y='Health Expenditure (% GDP)',
        color='Health Expenditure (% GDP)',
        title='Top 10 Countries Spending the Most on Public Health as % of GDP',
        labels={'Health Expenditure (% GDP)': 'Average Public Health Expenditure (% GDP)'},
        template='plotly_white',
        height=600,
        width=900,
    )
    return fig

def top10_health_expenditure(data):
    spending_data = data.groupby('Country Name')['Current Health Expenditure'].mean().reset_index()
    spending_data = spending_data.sort_values(by='Current Health Expenditure', ascending=False).head(10)

    # Create the bar plot
    fig = px.bar(
        spending_data,
        x='Country Name', y='Current Health Expenditure',
        color='Current Health Expenditure',
        title="Top 10 Countries by Per Capita Health Expenditure",
        labels={'Current Health Expenditure': 'Health Expenditure ($ PPP adjusted)'},
        template='plotly_white',
        color_continuous_scale='Reds',
    )

    # Customize layout
    fig.update_layout(
        height=600,
        width=900,
        xaxis_title="Country",
        yaxis_title="Health Expenditure",
        coloraxis_colorbar=dict(title="OOP Expenditure"),
    )
    return fig


# Streamlit App Layout
st.title("Public Health and Happiness Dashboard")

# sidebar option selection
st.sidebar.title("Dashboard Navigation")
options = st.sidebar.radio(
    "Select a Visualization:",
    [
        "Public Health Expenditure vs Life Expectancy",
        "Public Health Expenditure vs Happiness",
        "Happiness vs Life Expectancy",
        "Life Expectancy vs Health Expenditure and Happiness",
        "OOP Expenditure vs Current Health Expenditure",
        "Over Time Line Chart", 
        "Top 10 Countries by Public Health Expenditure as % of GDP",
        "Top 10 Countries by Per Capita Health Expenditure",
    ]
)

# Sidebar filters
st.sidebar.title("Dashboard Filters")

select_all = st.sidebar.checkbox("Select All Countries", value=True)

default_countries = ['United Kingdom', 'Japan', 'Denmark', 'Australia', 'India', 'Germany', 'China', 'Brazil', 'Spain', 'Argentina']
available_countries = merged_data['Country Name'].unique()
if select_all:
    selected_countries = available_countries.tolist()
else:
    selected_countries = st.sidebar.multiselect(
        "Select Countries to Display",
        options=available_countries,
        default=default_countries,
    )

selected_years = st.sidebar.slider("Select Year Range", min_value=int(merged_data['Year'].min()), max_value=int(merged_data['Year'].max()), value=(2015, 2021))


if len(selected_countries) < 1:
    st.warning(f"Please select at least 1 option. (Displaying all data by defalt)")
elif selected_countries: #check if list is not empty
    selected_countries_string = ", ".join(selected_countries)
    st.write(f"\n\n\n\n **Showing results for :** {selected_countries_string}")
else:
    st.write("Please select options.")

if len(selected_countries) >= 1:
    filtered_data = merged_data[merged_data['Country Name'].isin(selected_countries)]
else: 
    filtered_data = merged_data[merged_data['Country Name'].isin(default_countries)]

filtered_data = filtered_data[(filtered_data['Year'] >= selected_years[0]) & (filtered_data['Year'] <= selected_years[1])]

if options == "Public Health Expenditure vs Life Expectancy":
    st.plotly_chart(PHEvsLE_plotly(filtered_data), use_container_width=True)

elif options == "Public Health Expenditure vs Happiness":
    st.plotly_chart(PHEvsH_plotly(filtered_data), use_container_width=True)

elif options == "Happiness vs Life Expectancy":
    st.plotly_chart(HvsLE_plotly(filtered_data), use_container_width=True)

elif options == "OOP Expenditure vs Current Health Expenditure":
    st.plotly_chart(OOPEvsCHE_plotly(filtered_data), use_container_width=True)
    st.markdown("""
                **Note:**  
                - **OOP (Out-of-Pocket Expenditure):** The direct payments made by individuals to healthcare providers.  
                - **Current Health Expenditure:** Total health expenditure including OOP and government contributions as a share of GDP.  
                """)

elif options == "Life Expectancy vs Health Expenditure and Happiness":
    st.plotly_chart(HvsLEvsHE_plotly(filtered_data), use_container_width=True)

elif options == "Over Time Line Chart":
    over_time_line_chart(filtered_data)

elif options == "Top 10 Countries by Public Health Expenditure as % of GDP":
    st.plotly_chart(top10_public_health_spending(merged_data), use_container_width=True)

elif options == "Top 10 Countries by Per Capita Health Expenditure":
    st.plotly_chart(top10_health_expenditure(merged_data), use_container_width=True)

st.sidebar.info(
    """
    This dashboard explores relationships between public health expenditure, 
    life expectancy, and happiness metrics across selected countries.
    """
)
st.sidebar.download_button(label="Download Filtered Data", data=filtered_data.to_csv(index=False), file_name='filtered_data.csv', mime='text/csv',)
