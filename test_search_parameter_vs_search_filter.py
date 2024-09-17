# %%
from pyalex import Authors, Works

# %% [markdown]
# ## Check if we can look up alternatives of author's diplay names
# 
# Open Alex offers both the [search parameter](https://docs.openalex.org/api-entities/authors/search-authors) and the [search filter](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists) for this. Pyalex supports both, but the search parameter is not really documentented in their [documentation](https://pypi.org/project/pyalex/#description).
# 
# According to Open Alex, both variants should essentially do the same, so let's test it.
# 
# First, let's define a couple of variants on how to spell his name.

# %%
display_name = "Carl Sagan"
alternative_display_name = "Carl Edward Sagan"
lowercase_display_name = display_name.lower()
lowercase_alternative_display_name = alternative_display_name.lower()

# %% [markdown]
# We can see that the `display_name` is the default, where as the `alternative_display_name` is one alternative spelling for him.

# %%
print(Authors().search_filter(display_name=display_name).get()[0]['id'])
print(Authors().search_filter(display_name=display_name).get()[0]['display_name'])
print(Authors().search_filter(display_name=display_name).get()[0]['display_name_alternatives'])

# %% [markdown]
# However, if we search search for the alternative name with `search_filter()`, we get a different author!

# %%
print(Authors().search_filter(display_name=alternative_display_name).get()[0]['id'])
print(Authors().search_filter(display_name=alternative_display_name).get()[0]['display_name'])
print(Authors().search_filter(display_name=alternative_display_name).get()[0]['display_name_alternatives'])


# %% [markdown]
# `search_filter()` does not seemt to be case-sensitve though!

# %%
print(Authors().search_filter(display_name=lowercase_display_name).get()[0]['id'])
print(Authors().search_filter(display_name=lowercase_display_name).get()[0]['display_name'])
print(Authors().search_filter(display_name=lowercase_display_name).get()[0]['display_name_alternatives'])

# %% [markdown]
# However, with the search parameter, we can search for an alternative display name as expected and documented.
# 

# %%
print(f"Searching for {display_name}")
print(Authors().search(display_name).get()[0]['id'])
print(Authors().search(display_name).get()[0]['display_name'])
print(Authors().search(display_name).get()[0]['display_name_alternatives'])

print("")

print(f"Searching for {alternative_display_name}")
print(Authors().search(alternative_display_name).get()[0]['id'])
print(Authors().search(alternative_display_name).get()[0]['display_name'])
print(Authors().search(alternative_display_name).get()[0]['display_name_alternatives'])

# %% [markdown]
# search() is also not case-sensitive, which is nice!

# %%
print(f"Searching for {lowercase_display_name}")
print(Authors().search(lowercase_display_name).get()[0]['id'])
print(Authors().search(lowercase_display_name).get()[0]['display_name'])
print(Authors().search(lowercase_display_name).get()[0]['display_name_alternatives'])

print("")

print(f"Searching for {lowercase_alternative_display_name}")
print(Authors().search(lowercase_alternative_display_name).get()[0]['id'])
print(Authors().search(lowercase_alternative_display_name).get()[0]['display_name'])
print(Authors().search(lowercase_alternative_display_name).get()[0]['display_name_alternatives'])

# %% [markdown]
# Let's double check for an author that we determined at random.

# %%
print(Authors()["A5086799468'"]['display_name']) # 'Olga VIZITIU'
print(Authors()["A5086799468'"]['display_name_alternatives']) # 'Olga VIZITIU'

# %%
print(f"Searching for Olga VIZITIU")
print(Authors().search('Olga VIZITIU').get()[0]['id'])
print(Authors().search('Olga VIZITIU').get()[0]['display_name'])
print(Authors().search('Olga VIZITIU').get()[0]['display_name_alternatives'])

print("")

print(f"Searching for Gaţe, O.P.")
print(Authors().search("Gaţe, O.P.").get()[0]['id'])
print(Authors().search("Gaţe, O.P.").get()[0]['display_name'])
print(Authors().search("Gaţe, O.P.").get()[0]['display_name_alternatives'])

# %% [markdown]
# We should check if this is an Open Alex problem or a pyalex problem and report it in an issue.
# 
# How can we get the search query by the way, so we can test what urls we are actually using.

# %%
print(f"Searching for Tamarinde Haven")

author = Authors().search('T. Haven').get()[0]

#display_name_author = author['id']

print(Authors().search('T. Haven').get()[0]['id'])
print(Authors().search('Tamarinde Haven').get()[0]['display_name'])
print(Authors().search('Tamarinde Haven').get()[0]['display_name_alternatives'])

# %%
Authors()["A5066895944"]['affiliations']

# %%
Works().search('towards a responsible research climate').get()[0]['authorships']

# %%
print(Authors().search('Afke Ekels').get()[0]['id'])

# %%
Authors()["A5064823029"]['affiliations']

# %%
print(Authors().search('Stefan Kirsch').get())

# %% [markdown]
# print(Authors().search('Stefan Kirsch').get()['id'])


