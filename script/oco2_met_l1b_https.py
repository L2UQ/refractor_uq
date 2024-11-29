import earthaccess

# Orbit Select
grnstr = '05862a'

# Find L2Met
short_name = 'OCO2_L2_Met'
version = '11r'
start_time = '2015-08-08'
end_time = '2015-08-09'

results = earthaccess.search_data(
    short_name=short_name,
    version=version,
    cloud_hosted=True,
    temporal=(start_time,end_time)
)

for j in range(len(results)):
    crlnk = results[j]
    lnkstr = crlnk.data_links()[0]
    if (grnstr in lnkstr):
        metlnk = results[j]
        print(lnkstr)

#urls_l2met = [granule.data_links()[0] for granule in results]
#print(urls_l2met)

# Find L1bSc
short_name = 'OCO2_L1B_Science'
version = '11r'
start_time = '2015-08-08'
end_time = '2015-08-09'

results = earthaccess.search_data(
    short_name=short_name,
    version=version,
    cloud_hosted=True,
    temporal=(start_time,end_time)
)

for j in range(len(results)):
    crlnk = results[j]
    lnkstr = crlnk.data_links()[0]
    if (grnstr in lnkstr):
        l1blnk = results[j]
        print(lnkstr)

# Download
dnld_met = earthaccess.download(metlnk,local_path='./test_data')
dnld_l1b = earthaccess.download(l1blnk,local_path='./test_data')
        