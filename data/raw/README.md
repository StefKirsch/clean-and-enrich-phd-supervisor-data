# Raw data

The project is based on NARCIS (National Academic Research and Collaboration Information System) harvested meta-data from different PhD thesis repositories in The Netherlands (NARCIS, n.d.). NARCIS aggregates the information on scholarly work across Dutch academic institutions. NARCIS followed standardized protocols during the extraction, ensuring consistency and interoperability across various data sources.

By now the NARCIS dataset has been included into OpenAIRE.

To obtain the original dataset, please contact @tamarinde on Github.

The raw dataset should be placed in `data/raw/pairs_sups_phds.csv`. Make sure that the columns have the following names: `thesis_identifier`,`contributor`,`contributor_order`,`institution`,`author_name`,`title`,`year`,`language`