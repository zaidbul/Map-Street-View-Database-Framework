# Map-Street-View-Database-Framework

---

Framework for building Street Level View datasets quickly and seamlessly, labeled
and with Coordinate Metadata.

Used in project: https://github.com/zaidbul/Image-Geolocation-Framework-via-Deep-Learning

---

---

# CURRENT VER:

## API RELIANT

## Bugs:
- Method for pause/resuming downloads needs to be adjusted, current method does
not take into account possible interference
  - Possible outlook for fixes:
    - Lower total CSV files from 3 to 2, currently unnecessarily redundant
    - Initiate script by current image count / names / hashes
      - Items that are logged and NOT FLAGGED AS DUPLICATE but do not show up
      in new count will be automatically reinstalled
    - During deduplication, make sure recounting / updating files is constant

## Features that need to be added:
- **Cost calculator**
  - Predict current costs and ask user if they would want to move forward
    - Have feature to predict costs dynamically based on if they are just starting
    or if they have a backlog of images already
    - Scrape google cost estimate to see current estimates, and then dynamically
    calculate the predicted MAX cost per full run (Price based on granularity
    set by user, as well as headings)

- #### **Available granularity**
  - The issue of duplicates or non-existing images arises from uncertainty in
  the granularity of the data that google holds, if there exists a method to know beforehand
  or use ```MAX_GRANULARITY``` to stop at the upper limit, this would be an extra step for
  positive redundancy / cost saving.
    - Might be more useful on [API INDEPENDENT](#api-independent---web-scraping-practice)

---

# FUTURE VER

## API INDEPENDENT - WEB SCRAPING PRACTICE

Self learning project on web scraping / data mining

!! FOR LEARNING PURPOSES ONLY !!

 - Ideas:
   - Remove constant hash checking against entire database, instead have single ```undefinded``` hash from googles
   error output which will be checked against
   - Figure out method of download, if using a browser:
     - Could we use multiple browsers?
     - Many instances downloading at once to complete a download?
     - Using the area MIN/MAX LON/LAT , we could hold a map download progress
     and pause/resume from areas we left off (i.e. keep check of direction and end goal as well)
   - No self cost meaning we can work with duplicates if granularity is too specific,
   but check with Available Granularity feature above as it could remove possible storage cost