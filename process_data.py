#! /usr/bin/env python

import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import click
from configparser import ConfigParser

cnx_dir = os.getenv("CONNECTION_INFO")
parser = ConfigParser()
parser.read(os.path.join(cnx_dir, "db_conn.ini"))
psql_params = {k:v for k,v in parser._sections["disorientation"].items()}
psql_string = "postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(psql_string.format(**psql_params))


def process_permits():
    os.chdir('/home/nate/dropbox-caeser/Data/DPD_LYDIA/memphis_30')
    col = lambda x: [col.lower() for col in x.columns] 
    const = pd.read_csv('./data/new_construction_permits.csv')
    contr = pd.read_csv('./data/contractor_permits.csv')
    perm = pd.read_csv('./data/permits.csv')
    const.columns = col(const)
    contr.columns = col(contr)
    perm.columns = col(perm)
    col_diff = lambda x, y: set(x.columns).difference(set(y.columns))
    drop_cols = ['loc_name', 'status', 'score', 'match_type', 
                    'match_addr', 'side', 'addr_type', 'arc_street']

    for df in [const, contr, perm]:
        df.drop(drop_cols, axis=1, inplace=True)
        df['sub_type'] = df['sub_type'].str.lower()
        df['const_type'] = df['const_type'].str.lower()

    perm['year'] = perm.issued.str.split('/').str[0]
    comb = const.append(contr, ignore_index=True)
    comb.year = comb.issued.str[:4]
    comb = comb.append(perm, ignore_index=True)
    comb['dup'] = comb.duplicated([col for col in comb.columns])
    uni = comb[comb.dup == False]

    uni.replace({'descriptio':{'\r\n\r\n':' ', '\r\n':' '},
                'fraction':{'MEMP':'Memphis','CNTY':'Memphis','LKLD':'Lakeland',
                            'ARLI':'Arlington', 'MILL':'Millington',np.nan:'Memphis',
                            'BART':'Bartlett','CNY':'Memphis', 'CMTY':'Memphis',
                            'COLL':'Collierville','GTWN':'Germantown', 
                            '`':'Memphis'}}, inplace=True, regex=True)
    #replace didn't work when run as part of first replace statement
    #regex for 4 or more whitespace chars followed by any number of non-whitespace
    #chars followed by any character except line endings
    uni.address.replace('\s{4,}[\S]+.*','',inplace=True, regex=True)
    uni['state'] = 'TN'

    #New Construction
    uni['month'] = uni.issued.str[-5:7]
    new_cons = uni[uni.const_type == 'new']
    demos = uni[uni.const_type == 'demo']

def build_all_features():
    """Builds the all_features table which contains all features from the WWL
    dataset as well as a few other variables.
    """
    sql =("drop table if exists all_features;"
        "create table all_features as "
        "select w.geoid10, numpermit, numdemo, ninter/sqmiland inter_density,"
        "totpop/sqmiland popdensity, hhinc, "
        #"pct_wh,pct_bl, "
        "pct_bl + pct_ai + pct_as + "
        "pct_nh + pct_ot + pct_2m pct_nonwh, "
        "pct_pov_tot, pct_to14 + pct_15to19 pct_u19,"
        "pct_20to24, pct_25to34, pct_35to49, pct_50to66, "
        "pct_67up, hsng_density, pct_comm, age_comm, "
        "pct_dev, pct_vac, park_dist, park_pcap, gwy_sqmi, "
        "age_bldg, mdnhprice,mdngrrent, pct_afford, "
        "pct_hu_vcnt, affhsgreen, foreclose,pct_own, "
        "pct_rent, pct_mf, age_sf, mdn_yr_lived, "
        "strtsdw_pct, bic_index,"
        "b08303002 + b08303003 + b08303004 tt_less15,"
        "b08303005 + b08303006 + b08303007 tt_15to29,"
        "b08303008 + b08303009 + b08303010 + b08303011 "
        "+ b08303012 + b08303013 tt30more,"
        "b08301002 tm_caralone, b08301010 tm_transit, "
        "b08301018 tm_bicycle, b08301019 tm_walk, mmcnxpsmi, "
        "transit_access, bic_sqmi, rider_sqmi, vmt_per_hh_ami, "
        "walkscore, autos_per_hh_ami, pct_canopy, "
        "green_bldgs_sqmi, pct_chgprop, avg_hours, "
        "emp_ovrll_ndx, pct_labor_force, emp_ndx, pct_unemp, "
        "pct_commercial, pct_arts, pct_health, pct_other, "
        "pct_pubadmin, pct_util, pct_mining, pct_ag, "
        "pct_food, pct_retail, pct_wholesale, pct_manuf, "
        "pct_construction, pct_waste_mgmt, pct_ed, pct_info, "
        "pct_transport, pct_finance, pct_realestate, "
        "pct_prof_services, pct_mgmt,pct_lowinc_job, "
        "pct_b15003016 pct_no_dip, pct_b15003017 pct_dip, "
        "pct_b15003018 pct_ged, pct_b15003019 pct_uni_1yr, "
        "pct_b15003020 pct_uni_no_deg, pct_b15003021 pct_assoc, "
        "pct_b15003022 pct_bach, pct_b15003023 pct_mast, "
        "pct_b15003024 pct_prof_deg, pct_b15003025 pct_phd, "
        "elem_dist, middle_dist, high_dist, "
        "pvt_dist, chldcntr_dist, cmgrdn_dist, frmrmkt_dist, "
        "library_dist, commcenter_dist,pct_medicaid, "
        "bpinc_pcap, hosp_dist, pol_dist, fire_dist, "
        "os_sqmi, pct_imp, wetland_sqmi, brnfld_sqmi, "
        "mata_route_sqmi, mata_stop_sqmi "
    "from (select count(s.fid) ninter, t.wkb_geometry, geoid10 "
            "from tiger_tract_2010 t, streets_carto_intersections s "
            "where st_intersects(s.wkb_geometry, t.wkb_geometry) "
            "group by geoid10, t.wkb_geometry) bg, "
            "(select geoid10, "
            "count(distinct case when const_type = 'new' "
                "then permit end) numpermit, "
            "count(distinct case when const_type = 'demo' "
                "then permit end) numdemo "
            "from permits p, tiger_tract_2010 t "
            "where st_within(p.wkb_geometry, t.wkb_geometry) "
            "group by t.geoid10) p, "
            "wwl_2017_tract w "
    "where w.geoid10 = bg.geoid10 "
    "and w.geoid10 = p.geoid10;")  
    engine.execute(sql)

#def select_features(df):

@click.command()
@click.option("--all_features", '-af', is_flag=True,
    help="Build the all_features table")
def main(all_features):
    if all_features:
        build_all_features()


if __name__=="__main__":
    main()
