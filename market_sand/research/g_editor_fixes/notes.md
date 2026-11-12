# Editor's fixes

Small corrections made while merging the research, each with its source. They sit in their own folder so the
researchers' files stay as they delivered them.

- **Imperial Tobacco / Imperial Brands are two companies.** The original (1901) was taken over by Hanson Trust in
  April 1986 for about GBP 2.5bn (Bank of England Quarterly Bulletin 1989, "Takeover activity in the 1980s"); the
  Imperial Tobacco Group demerged by Hanson in 1996 (Imperial Brands since 2016) is a new company. Its rows from
  2012 on, delivered under `imperial_tobacco`, are moved to `imperial_brands` by `build_data.ID_SPLIT`. The old
  company's values stop at 1950 in the research; its pile moves smoothly from 1950 to the 1986 takeover price.
- **US Steel, 2001.** The island follows the steel company, not the legal heir (USX became Marathon Oil, with about
  85% of the value). Decided for clarity: viewers know US Steel; Marathon was never a top company. The 2001
  drop is marked as a spin-off.
