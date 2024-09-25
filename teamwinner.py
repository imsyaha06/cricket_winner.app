import streamlit as st
import pandas as pd
import joblib
import random

# Load the trained Decision Tree and Random Forest models and column transformer
dt_model = joblib.load('best_decision_tree_model.pkl')
rf_model = joblib.load('best_rf_model_compressed.joblib')  # Load the Random Forest model
ct = joblib.load('column_transformer.pkl')


teams = ['team1', 'team2']
# Randomly decide the toss winner
toss_winner = random.choice(teams)



# Function to make predictions
def predict_winner(team1, team2, venue):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'team1': [team1],
        'team2': [team2],
        'venue': [venue],
        'id': [0],  # Default values for missing columns
        'date': ['2023-01-01'],  # Placeholder date
        'dl_applied': [0],  # Assume D/L method is not applied
        'result': ['normal'],  # Placeholder result
        'toss_win': [toss_winner],  # Assume team1 won the toss
        'umpire3': [None],  # Assuming some columns might not be available
        'city': ['Unknown'],  # Placeholder city
        'player_of_match': [None],  # Placeholder
        'win_by_wickets': [0],  # Placeholder
        'umpire2': [None],  # Placeholder
        'toss_decision': ['bat'],  # Assume the toss decision is 'bat'
        'umpire1': [None],  # Placeholder
        'win_by_runs': [0],  # Placeholder
        'season': ['2023']  # Placeholder season
    })
    
    # Apply the column transformer to encode categorical features
    input_encoded = ct.transform(input_data)

    # Make predictions using both models
    dt_prediction = dt_model.predict(input_encoded)[0]
    rf_prediction = rf_model.predict(input_encoded)[0]

    return dt_prediction, rf_prediction

# Streamlit UI
st.title("ODI Match Winner Prediction")
st.write("Enter the details of the match to predict the winner:")

# User inputs
team1 = st.selectbox("Select Team 1", options=["Afghanistan", "Africa XI", "Asia XI", "Australia", "Bangladesh", "Bermuda", "Canada", 
                                               "England", "Hong Kong", "India", "Ireland", "Jersey", "Kenya", 
                                               "Namibia", "Nepal", "Netherlands", "New Zealand", "Oman", "Pakistan", 
                                               "Papua New Guinea", "Scotland", "South Africa", "Sri Lanka", "United Arab Emirates", 
                                               "United States of America", "West Indies", "Zimbabwe"])

team2 = st.selectbox("Select Team 2", options=["Afghanistan", "Africa XI", "Asia XI", "Australia", "Bangladesh", "Bermuda", "Canada", 
                                               "England", "Hong Kong", "India", "Ireland", "Jersey", "Kenya", 
                                               "Namibia", "Nepal", "Netherlands", "New Zealand", "Oman", "Pakistan", 
                                               "Papua New Guinea", "Scotland", "South Africa", "Sri Lanka", "United Arab Emirates", 
                                               "United States of America", "West Indies", "Zimbabwe"])   

venue = st.selectbox("Select Venue", options=["AMI Stadium", "Adelaide Oval", "Affies Park", "Al Amerat Cricket Ground Oman Cricket (Ministry Turf 1)",	"Al Amerat Cricket Ground Oman Cricket (Ministry Turf 2)", "Amini Park", "Port Moresby",
                                              "Andhra Cricket Association-Visakhapatnam District Cricket Association Stadium",	"Antigua Recreation Ground, St John's",	"Arbab Niaz Stadium", "Arnos Vale Ground",	"Arnos Vale Ground, Kingstown", 
                                              "Arnos Vale Ground, Kingstown, St Vincent", "Arun Jaitley Stadium", "Arun Jaitley Stadium, Delhi",	"Bangabandhu National Stadium",	"Bangabandhu National Stadium, Dhaka",	"Barabati Stadium",	"Barabati Stadium, Cuttack",
                                              "Barsapara Cricket Stadium",	"Barsapara Cricket Stadium, Guwahati",	"Basin Reserve", "Bay Oval", "Bay Oval, Mount Maunganui", "Beausejour Stadium, Gros Islet",	"Bellerive Oval",	"Bellerive Oval, Hobart", "Bert Sutcliffe Oval",	
                                              "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium", "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",	"Boland Bank Park, Paarl",	"Boland Park",	"Boland Park, Paarl",	"Brabourne Stadium",
                                              "Bready Cricket Club, Magheramason",	"Brian Lara Stadium, Tarouba, Trinidad", "Brisbane Cricket Ground",	"Brisbane Cricket Ground, Woolloongabba",	"Brisbane Cricket Ground, Woolloongabba, Brisbane",	"Buffalo Park",	"Buffalo Park", 
                                              "East London", "Bulawayo Athletic Club",	"Bundaberg Rum Stadium, Cairns",	"Cambusdoon New Ground",	"Cambusdoon New Ground, Ayr",	"Captain Roop Singh Stadium", "Captain Roop Singh Stadium, Gwalior", "Carisbrook",	"Castle Avenue", "Cazaly's Stadium, Cairns",
                                              "Central Broward Regional Park Stadium Turf Ground",	"Chevrolet Park",	"Chittagong Divisional Stadium",	"Choice Moosa Stadium, Pearland",	"City Oval, Pietermaritzburg",	"Civil Service Cricket Club, Stormont",	"Civil Service Cricket Club, Stormont, Belfast",
                                              "Clontarf Cricket Club Ground",	"Clontarf Cricket Club Ground, Dublin",	"Cobham Oval (New)", "County Ground", "County Ground, Bristol",	"County Ground, Chelmsford", "Daren Sammy National Cricket Stadium",
                                              "Daren Sammy National Cricket Stadium, Gros Islet",	"Darren Sammy National Cricket Stadium, Gros Islet",	"venue_Darren Sammy National Cricket Stadium, St Lucia",	"Davies Park, Queenstown",	
                                              "De Beers Diamond Oval",	"De Beers Diamond Oval, Kimberley",	"Diamond Oval",	"Diamond Oval, Kimberley",	"Docklands Stadium", "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",	"Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
                                              "Dubai International Cricket Stadium",	"Dubai Sports City Cricket Stadium", "Eden Gardens", "Eden Gardens, Kolkata",	"Eden Park", "Eden Park, Auckland",	"Edgbaston", "Edgbaston, Birmingham", "Feroz Shah Kotla",	"Gaddafi Stadium",	"Gaddafi Stadium, Lahore",
                                              "Galle International Stadium",	"Goodyear Park", "Goodyear Park, Bloemfontein",	"Grange Cricket Club Ground, Raeburn Place", "Grange Cricket Club Ground, Raeburn Place, Edinburgh",	"Grange Cricket Club, Raeburn Place",	"Greater Noida Sports Complex Ground",
                                              "Green Park", "Greenfield International Stadium",	"Greenfield International Stadium, Thiruvananthapuram",	"Gymkhana Club Ground",	"Gymkhana Club Ground, Nairobi",	"Hagley Oval",	"Hagley Oval, Christchurch", "Harare Sports Club",	"Hazelaarweg, Rotterdam", "Headingley",
                                               "Headingley, Leeds",	"Himachal Pradesh Cricket Association Stadium",	"Holkar Cricket Stadium", "Holkar Cricket Stadium, Indore", "ICC Academy",	"ICC Academy, Dubai",	"ICC Global Cricket Academy", "Indian Petrochemicals Corporation Limited Sports Complex Ground",
                                               "Iqbal Stadium",	"Iqbal Stadium, Faisalabad",	"JSCA International Stadium Complex",	"JSCA International Stadium Complex, Ranchi",	"Jade Stadium",	"Jade Stadium, Christchurch",	"Jaffery Sports Club Ground", "John Davies Oval",
                                               "Keenan Stadium",	"Kennington Oval",	"Kennington Oval, London",	"Kensington Oval, Barbados",	"Kensington Oval, Bridgetown", "Kensington Oval, Bridgetown, Barbados",	"Khan Shaheb Osman Ali Stadium", "Kingsmead", "Kingsmead, Durban",	
                                               "Kinrara Academy Oval",	"Lal Bahadur Shastri Stadium, Hyderabad, Deccan	venue_Lord's",	"Lord's, London",	"M Chinnaswamy Stadium", "M.Chinnaswamy Stadium",	"MA Aziz Stadium",	"MA Aziz Stadium, Chittagong",	"MA Chidambaram Stadium, Chepauk",
                                               "MA Chidambaram Stadium, Chepauk, Chennai",	"Madhavrao Scindia Cricket Ground",	"Maharani Usharaje Trust Cricket Ground",	"Maharashtra Cricket Association Stadium",	"Mahinda Rajapaksa International Cricket Stadium, Sooriyawewa",	"Mahinda Rajapaksa International Cricket Stadium, Sooriyawewa, Hambantota",
                                               "Malahide", "Mangaung Oval",	"Mangaung Oval, Bloemfontein",	"Mannofield Park",	"Mannofield Park, Aberdeen",	"Manuka Oval",	"Maple Leaf North-West Ground",	"Marrara Cricket Ground",	"Marrara Cricket Ground, Darwin", "McLean Park",	"McLean Park, Napier",	"Melbourne Cricket Ground",
                                               "Mission Road Ground, Mong Kok",	"Mombasa Sports Club Ground",	"Moosa Cricket Stadium, Pearland",	"Mulpani Cricket Ground",	"Multan Cricket Stadium",	"Nahar Singh Stadium",	"Nahar Singh Stadium, Faridabad",	"Narayanganj Osmani Stadium",	"Narendra Modi Stadium, Ahmedabad",	"National Cricket Stadium",	
                                               "National Cricket Stadium, Grenada",	"National Cricket Stadium, St George's", "National Stadium	venue_National Stadium, Karachi", "Nehru Stadium",	"Nehru Stadium, Fatorda",	"Nehru Stadium, Poona",	"New Wanderers Stadium",	"New Wanderers Stadium, Johannesburg",	"Newlands"	"Newlands, Cape Town"	"Niaz Stadium, Hyderabad",
                                               "North West Cricket Stadium, Potchefstroom",	"OUTsurance Oval",	"Old Hararians",	"Old Trafford",	"Old Trafford, Manchester",	"P Saravanamuttu Stadium",	"Pallekele International Cricket Stadium",	"Perth Stadium", "Providence Stadium",	"Providence Stadium, Guyana", "Punjab Cricket Association IS Bindra Stadium, Mohali",	
                                               "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",	"Punjab Cricket Association Stadium, Mohali",	"Queen's Park (New), St George's, Grenada",	"Queen's Park Oval", "Queen's Park Oval, Port of Spain", "Queen's Park Oval, Port of Spain, Trinidad",	"Queen's Park Oval, Trinidad", 
                                               "Queens Sports Club",	"Queens Sports Club, Bulawayo",	"Queenstown Events Centre",	"R Premadasa Stadium",	"R Premadasa Stadium, Colombo",	"R.Premadasa Stadium",	"R.Premadasa Stadium, Khettarama",	"Rajiv Gandhi International Cricket Stadium, Dehradun",	"Rajiv Gandhi International Stadium, Uppal",
                                               "Rajiv Gandhi International Stadium, Uppal, Hyderabad",	"Rangiri Dambulla International Stadium",	"Rawalpindi Cricket Stadium",	"Reliance Stadium",	"Riverside Ground",	"Riverside Ground, Chester-le-Street",	"Riverway Stadium, Townsville",	
                                               "Ruaraka Sports Club Ground"	"Sabina Park, Kingston",	"Sabina Park, Kingston, Jamaica", "Sardar Patel (Gujarat) Stadium, Motera",	"Sardar Patel Stadium, Motera",	"Saurashtra Cricket Association Stadium",	"Sawai Mansingh Stadium",	"Saxton Oval",	"Sector 16 Stadium",	"Seddon Park",	"Seddon Park, Hamilton", "Sedgars Park",	
                                               "Sedgars Park, Potchefstroom",	"Senwes Park",	"Senwes Park, Potchefstroom",	"Shaheed Chandu Stadium",	"Shaheed Veer Narayan Singh International Stadium, Raipur",	"Sharjah Cricket Association Stadium",	"Sharjah Cricket Stadium",	"Sheikh Abu Naser Stadium",	"Sheikh Zayed Stadium",	"Sheikhupura Stadium",	"Sher-e-Bangla National Cricket Stadium	",
                                               "Shere Bangla National Stadium",	"Shere Bangla National Stadium, Mirpur",	"Sinhalese Sports Club",	"Sinhalese Sports Club Ground",	"Sir Vivian Richards Stadium",	"Sir Vivian Richards Stadium, North Sound",	"Sophia Gardens",	"Sophia Gardens, Cardiff",	"Sportpark Het Schootsveld",	"Sportpark Maarschalkerweerd, Utrecht",	
                                               "St George's Park",	"St George's Park, Port Elizabeth",	"St Lawrence Ground",	"St Lawrence Ground, Canterbury",	"SuperSport Park",	"SuperSport Park, Centurion",	"Sydney Cricket Ground",	"Sylhet International Cricket Stadium",	"Takashinga Sports Club, Highfield, Harare",	"The Cooper Associates County Ground",	"The Rose Bowl",	"The Rose Bowl, Southampton	",
                                               "The Royal & Sun Alliance County Ground, Bristol",	"The Village, Malahide",	"The Village, Malahide, Dublin",	"The Wanderers Stadium",	"The Wanderers Stadium, Johannesburg",	"Titwood",	"Titwood, Glasgow",	"Tony Ireland Stadium, Townsville",	"Toronto Cricket, Skating and Curling Club",	"Trent Bridge",	"Trent Bridge, Nottingham",	
                                               "Tribhuvan University International Cricket Ground",	"Tribhuvan University International Cricket Ground, Kirtipur",	"United Cricket Club Ground, Windhoek",	"University Oval",	"VRA Cricket Ground",	"VRA Ground",	"VRA Ground, Amstelveen",	"Vidarbha C.A. Ground",	"Vidarbha Cricket Association Ground",	"Vidarbha Cricket Association Stadium, Jamtha",	"W.A.C.A. Ground",	
                                               "Wanderers Cricket Ground",	"Wanderers Cricket Ground, Windhoek",	"Wankhede Stadium",	"Wankhede Stadium, Mumbai",	"Warner Park, Basseterre",	"West End Park International Cricket Stadium, Doha",	"Western Australia Cricket Association Ground",	"Westpac Park, Hamilton",	"Westpac Stadium",	"Westpac Stadium, Wellington",	"Willowmoore Park",	"Willowmoore Park, Benoni",	
                                               "Windsor Park, Roseau",	"Zahur Ahmed Chowdhury Stadium",	"Zahur Ahmed Chowdhury Stadium, Chattogram",	"Zohur Ahmed Chowdhury Stadium",
])  # Replace with actual venues

# Prediction
if st.button("Predict Winner"):
    if team1 == team2:
        st.error("Team 1 and Team 2 cannot be the same!")
    else:
        dt_winner, rf_winner = predict_winner(team1, team2, venue)
        st.success(f"The predicted winner by Decision Tree is: {dt_winner}")
        st.success(f"The predicted winner by Random Forest is: {rf_winner}")
