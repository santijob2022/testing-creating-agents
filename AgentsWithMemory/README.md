# Agents with Memory

## Objective:

Build a personalized AI assistant that will handle your emails related to job offers and opportunities that come from your personal web page.

## AI-agent Characteristics:

The core of this project will be to set up and experiment with different type of memories. The memories will be of 3 types:

- Semantic memory: remember facts about the user. This is intended for user's personalization
- Episodic memory: this will be used to remember past actions in specific situations, for instance, changing the behavior with respect to a specific email sender.
- Procedural memory: able to modify the whole system state.

## Some Variants

- I started implementing a local database using qdrant.
This goal sets a series of challenges:
Currently langmem does not support vector databases like qdrant, so to be able to use create and search memory tools in langmem. I had to implement a BaseStore wrapper, so I can use qdrant along with langmem.

- I will add a second variant where I will use BaseStory. And everything will be way easier. I'm thinking about postgres which supports hybrid search, it's available local and on cloud, and is also compatible with langmem tools.