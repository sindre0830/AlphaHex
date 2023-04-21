# Import and initialize your own actor 
from ..source.state_manager import StateManager
from ..source.anet import ANET
actor = MyHexActor()
# Import and override the `handle_get_action` hook in ActorClient
from ActorClient import ActorClient 
class MyClient(ActorClient):
    def handle_get_action(self, state):
        row, col = actor.get_action(state)  # Your logic
        return row, col
        
# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient()
    client.run()
