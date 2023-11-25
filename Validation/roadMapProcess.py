from geopy.geocoders import Nominatim
import networkx as nx
G=nx.read_graphml("/Users/ksalamatian/CLionProjects/NewCurvature/roadnetwork4.graphml")
geolocator = Nominatim(user_agent="geoapiExercises")
for node in G.nodes():
    if "country" not in G.nodes[node]:
        Latitude=str(G.nodes[node]["lat"])
        Longitude=str(G.nodes[node]["long"])
        location = geolocator.reverse(Latitude+","+Longitude)
        address = location.raw['address']
        if "country_code" in address:
            G.nodes[node]["country"] = address['country_code']
        else:
            print(address)
            print(node, G.nodes[node]["meta"])

largest = max(nx.connected_components(G), key=len)
S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
nx.write_graphml(S[0],"/Users/ksalamatian/CLionProjects/NewCurvature/roadnetwork4.graphml")