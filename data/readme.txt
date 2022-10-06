The first line in each instance contains five numbers. In order, they have the following meaning:
- Number of vehicles
- Number of requests
- Maximum route duration (total duration of Vehicle's route cannot exceed this value)
- Vehicle capacity
- Maximum ride time (of a request)

After the first lines follows 2n+2 lines defining the nodes in the instance. Nodes 0 and 2n+1 correspond to the start
and end depot, nodes 1,...,n corresponds to pickup nodes and nodes n+1, ..., 2n corresponds to delivery nodes. Nodes
i and n+i forms a request (1 <= i <= n).

Each line defining a node contains the following data

<id> <x> <y> <service time> <demand> <TW start> <TW end>

where id is the node id, x and y are the coordinates of the node, demand is positive for pickups and negative for
deliveries, <TW start> and <TW end> define the start and end of the time window for the node.


Travel costs and travel times were calculated using the Euclidean distance function and were represented as double
precision floating point numbers.

source:
http://neumann.hec.ca/chairedistributique/data/darp/branch-and-cut/
