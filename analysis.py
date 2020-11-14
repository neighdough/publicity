#!/usr/bin/env python
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.manifold import TSNE
from collections import OrderedDict
from configparser import ConfigParser
from sqlalchemy import create_engine, text
import click
import json
import itertools
import re

cnx_dir = os.getenv("CONNECTION_INFO")
parser = ConfigParser()
parser.read(os.path.join(cnx_dir, "db_conn.ini"))
psql_params = {k:v for k,v in parser._sections["disorientation"].items()}
psql_string = "postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(psql_string.format(**psql_params))
pd.set_option('display.width', 180)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', 125)
home = os.getenv("HOME")
out_dir = os.path.join(home,"Dropbox/phd/dissertation/")
fig_home = os.path.join(home,out_dir, "figs")

df = pd.read_sql("select * from all_features", engine)
df.fillna(0, inplace=True)
x_vars = [col for col in df.columns if col not in 
            ['numpermit', 'numdemo', 'geoid10', 'wkb_geometry', 
             'scale_const', 'scale_demo', 'net']]
min_max_scale = lambda x: (x-x.min())/(x.max() - x.min())
std_scale = lambda x: (x-x.mean())/float(x.std())
df['scale_const'] = min_max_scale(df.numpermit)
df['scale_demo'] = min_max_scale(df.numdemo)
#permit column is actually net construction, but needs to be named permit
#to run correctly in fact_heatmap function
df['net'] = df.scale_const - df.scale_demo
RANDOM_STATE = 1416

#fig = px.parallel_coordinates(df[x_vars+["net"]], color="net", labels=x_vars+["net"], 
#color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=0)
#ply.offline.plot(fig)

def add_missing(df):
    idx_cols = ["month", "year", "name"]
    idx = list(itertools.product(*[range(1,13),
                                   range(2002,2017),
                                   df.name.unique()]))
    df.set_index(idx_cols, inplace=True)
    df = df.reindex(idx)
    df.reset_index(inplace=True)
    df.permit.fillna(0, inplace=True)
    return df

#-----------------------------------------------------------------------------
#--------------------------- Urban Index -------------------------------------
#-----------------------------------------------------------------------------
sql =("select w.geoid10, numpermit, numdemo, ninter/sqmiland inter_density,"
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

@click.group()
def main():
    pass

@main.command()
def corr_matrix():
    """Create correlation matrix and generate heatmap

    """
    from string import ascii_letters
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    X = df[x_vars]
    sns.set(style="white")

    # Compute the correlation matrix
    corr = X.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(22, 18))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}
                ,xticklabels=True, yticklabels=True
                )    
    ax.set_xticklabels(corr.index, fontsize=6)
    ax.set_yticklabels(corr.index, fontsize=6)
    plt.tight_layout()
    plt.savefig(out_dir+"/figs/corr_matrix.png")

@main.command()
@click.option("--coeff", "-c", default=.8)
@click.option("--update/--no-update", default=True)
def get_correlated_features(coeff, update):
    """Identify which features have highest correlation coefficient. The method
    returns a list of column names that should be excluded from an analysis,
    but also updates a table in the project database

    Args:
        coeff (float): threshold to identify values that should be removed

    """
    corr = df[x_vars].corr()
    #correlates = set()
    correlates = list()
    for col_idx in range(len(corr.columns)):
        for row_idx in range(col_idx):
            corr_coeff = corr.iloc[col_idx, row_idx]
            if  abs(corr_coeff) > coeff:
                d = {"variable": corr.columns[col_idx],
                     "corr_variable": corr.index[row_idx],
                     "corr_coeff": corr_coeff}
                correlates.append(d)
    df_corr = pd.read_json(json.dumps(correlates), orient="records")
    if update:
        df_corr.to_sql("correlation_values", con=engine, if_exists="replace", 
            index=False)
    return df_corr.variable.to_list()

def correlated_feature_list(coeff=.8):
    try:
        df_corr = pd.read_sql("select * from correlation_values", engine)
        return df_corr.variable.to_list()
    except:
        corr_feats = get_correlated_features(coeff, False)
        return corr_feats

@main.command()
@click.option("--yname", default="net")
@click.option("--n_features", "-nf", default=1)
@click.option("--plot/--no-plot", default=False)
@click.option("--coeff", "-c", default=.8)
def create_feature_scores(yname, n_features, plot, coeff):
    """Select most important features using recursive feature elimination (RFE)
        in conjunction with random forest regression and then plot the accuracy
        of the fit.

    References:

        Title: Feature Ranking RFE, Random Forest, Linear Models
        Author: Arthur Tok
        Date: June 18, 2018
        Code version: 80
        Availability: https://bit.ly/37ngDg8

        Title: Selecting good features – Part IV: stability selection, RFE and everything side by side
        Author: Ando Saabas
        Date: December 20, 2014
        Availability: https://bit.ly/2SGCuLx


    """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor 
    from sklearn.feature_selection import RFE

    corr_features = correlated_feature_list(coeff)
    X = df[[col for col in x_vars if col not in corr_features]]
    cols = X.columns
    feature_rank = {}
    y = df[yname]
    accuracy = []

    def rank_features(ranks, names, order=1):
        minmax = MinMaxScaler()
        feature_rank = minmax.fit_transform(order*np.array([ranks]).T).T[0]
        feature_rank = map(lambda x: round(x,2), feature_rank)
        return dict(zip(names, feature_rank))

    #********************* Recursive Feature Elimination ***********    
    #RFE with Linear Regression
    lr = LinearRegression(normalize=True)
    lr.fit(X,y)
    rfe = RFE(lr, n_features_to_select=n_features, verbose=3)
    rfe.fit(X,y)
    feature_rank["rfe-lr"] = rank_features(list(map(float, rfe.ranking_)), cols,
                        order=-1)
    accuracy.append(["rfe-lr", rfe.score(X,y)])    

    
    #RFE with Random Forest Regression
    rfr = RandomForestRegressor(max_features="sqrt", random_state=RANDOM_STATE)
    rfr.fit(X,y)
    rfe = RFE(rfr, n_features_to_select=n_features, verbose=3)
    rfe.fit(X,y)
    feature_rank["rfe-rfr"] = rank_features(list(map(float, rfe.ranking_)), cols,
                        order=-1)
    accuracy.append(["rfe-rfr", rfe.score(X,y)])

    #************************* Regression *****************************
    #Linear Regression alone
    lr = LinearRegression(normalize=True)
    lr.fit(X,y)
    feature_rank["lr"] = rank_features(np.abs(lr.coef_), cols)
    #Ridge Regression
    ridge = Ridge(alpha=7)
    ridge.fit(X,y)
    feature_rank["ridge"] = rank_features(np.abs(ridge.coef_), cols)
    accuracy.append(["ridge", ridge.score(X,y)])

    #Lasso
    lasso = Lasso(alpha=.05)
    lasso.fit(X,y)
    feature_rank["lasso"] = rank_features(np.abs(lasso.coef_), cols)
    accuracy.append(["lasso", lasso.score(X,y)])

    #Random Forest Regression alone
    rfr = RandomForestRegressor(max_features="sqrt", random_state=RANDOM_STATE)
    rfr.fit(X,y)
    feature_rank["rfr"] = rank_features(rfr.feature_importances_, cols)
    accuracy.append(["rfr", rfr.score(X,y)])

    r = {}
    for col in cols:
        r[col] = round(np.mean([feature_rank[method][col] 
                    for method in feature_rank.keys()]),2)
    methods = sorted(feature_rank.keys())
    feature_rank["mean"] = r
    df_feature_rank = pd.DataFrame.from_dict(feature_rank)
    df_feature_rank.to_sql("feature_rank_{}".format(yname),engine,
                            if_exists='replace')
    sort_feat_rank = df_feature_rank.sort_values("mean", ascending=False)
    sort_feat_rank["colnames"] = sort_feat_rank.index
    #plot feature rankings
    if plot:
        f, ax = plt.subplots(figsize=(34,22))
        f = sns.barplot(x="mean", y="colnames", data=sort_feat_rank,
                        palette="coolwarm")
        f.set_yticklabels(sort_feat_rank.index, fontsize=10)
        f.set_xlabel("Mean Feature Importance")
        f.set_ylabel("Column Name")
        f.figure.tight_layout(pad=6.)
#        f.fig.suptitle("Mean Feature Importance for {}".format(yname))
        plt.savefig(out_dir+"/figs/bar_feat_ranking_{}_horizontal.png".format(yname))
    return accuracy

@main.command()
@click.option("--coeff", "-c", default=.8)
@click.option("--plot/--no-plot", default=True)
@click.option("--yname", default="net")
@click.option("--n-features", "-nf", default=1)
@click.option("--feature-score", "-fc", default=0.)
def update_all(coeff, yname, n_features, feature_score, plot):
    """Update `correlation_values` and `feature_rank` tables and generate new
    plots for rfr_accuracy, bar_feat_ranking 
    """
    get_correlated_features(coeff, plot)
    create_feature_scores(yname, n_features, True, coeff)  
    plot_estimates(plot, feature_score, yname)

def select_features(yname="net", index_only=False, min_score=.0):
    """Select the feature rank table (feature_rank_<yname>) from postgres

    Args:
        yname (str): the suffix for which postgres table should be selected. 
            Accepted values are net, scale_const, or scale_demo. Defaults to 'net'
        index_only (bool): False if only the index column containing the column
            names should be returned, True if all columns from the table should
            be returned. Defaults to False
        min_score (float): the minimum mean importance score that should be returned.
            Defaults to .0 for all values.
    
    Returns:

    """
    cols = "index" if index_only else "*"
    params = {"yname": yname, "cols": cols, "mean": min_score}
    sql = "select {cols} from feature_rank_{yname} where mean >= {mean}"
    df = pd.read_sql(sql.format(**params), engine)
    return df

@main.command()
@click.option("-y", default="net")
@click.option("--all", "-a", is_flag=True)
def scatter_plot(y, all):
    """Generates scatter plot matrix for all predictor variables against a y-value
    such as net construction, total construction or toal demolition.

    Args:
        y (str): column name to be used for dependent variable
        all (bool): True if all features should be included, False if correlated
            variables should be excluded
    Returns:
        None
    """
    if not all:
        corr_cols = correlated_feature_list()
        cols = sorted([col for col in x_vars if col not in corr_cols])
    else:
        cols = sorted([col for col in x_vars])
    nrows, ncols = 9,14#14,9#10,12
    widths = [1 for col in range(ncols)]
    heights = [1 for row in range(nrows)]
    gridspec_dict={"width_ratios":widths,
                   "height_ratios": heights}
    f, axes = plt.subplots(nrows, ncols, sharex=False, sharey=True,
                    tight_layout=True, figsize=(34,22), 
                    gridspec_kw=gridspec_dict
                    )
    var_pos = 0
    def plot(var_pos, row, col):
        #if y-value is for net construction, add two plots, one for net-poitive
        #construction, the other for net negative
        aspect = "auto"
        if y == "net":
            df[df.net < 0].plot.scatter(x=cols[var_pos], y=y, marker="<",
                ax=axes[row,col],color="Purple")
            axes[row,col].set_aspect(aspect)
            df[df.net >= 0].plot.scatter(x=cols[var_pos], y=y, marker=">",
                ax=axes[row,col], color="Green")       
            axes[row,col].set_aspect(aspect)
        else:
            color = lambda x: "Green" if "const" in x else "Purple"
            df.plot.scatter(x=cols[var_pos], y=y, marker=">",
                ax=axes[row,col], color=color(y))
            axes[row,col].set_aspect(aspect)
    for row in range(nrows):
        for col in range(ncols):
            if var_pos < len(cols):
                plot(var_pos, row, col)
            var_pos += 1
    plt.savefig(out_dir+"/figs/scatter_plot_all_feats_{}.png".format(y))
    plt.close()

def scatter_plot_single(y, x_label, filter_val=None):
    if filter_val:
        new_df = df[df[x_label] <= filter_val]
    else:
        new_df = df.copy()
    colors = lambda x: "Purple" if x < 0 or "demo" in y.lower() else "Green"
    new_df["color"] = new_df[y].apply(colors)
    new_df.plot.scatter(x=x_label, y=y, color=new_df.color, s=8)
    plt.savefig(fig_home + "/scatter_plot_{0}_{1}.png".format(y,x_label))
    plt.close()

@main.command()
@click.option("--plot/--no-plot", default=False)
@click.option("--feat_score", "-fs", default=.25)
@click.option("--yname", "-y", default="net")
def plot_estimates(plot, feat_score, yname):
    """Generate plots comparing different different numbers of estimators to
    determine the number to use for final model

    """

    features = select_features(index_only=True, min_score=feat_score, yname=yname)
    #determinte number of trees in forest
    ensemble_clfs = [
        ("RFR, max_features='sqrt'|red|-",
            RandomForestRegressor(warm_start=True, oob_score=True,
                                max_features="sqrt",
                                random_state=RANDOM_STATE
                                )),
        ("RFR, max_features='log2'|green|-",
            RandomForestRegressor(warm_start=True, max_features='log2',
                                oob_score=True,
                                random_state=RANDOM_STATE
                                )),
        ("RFR, max_features=None|blue|-",
            RandomForestRegressor(warm_start=True, max_features=None,
                                oob_score=True,
                                random_state=RANDOM_STATE
                                ))]
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 15
    max_estimators = 500
    X = df[features["index"]]
    y_net = df[yname]
    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X, y_net)
            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        label, color, linestyle = label.split('|')
        plt.plot(xs, ys, label=label, color=color,
                linestyle=linestyle)

    if plot:
        plt.xlim(min_estimators, max_estimators)
        plt.xlabel("n_estimators")
        plt.ylabel("OOB error rate")
        #plt.legend(bbox_to_anchor=(0, 1.1, 1., .102), loc="upper center", ncol=2)
        plt.legend(ncol=2)
        title = ("Estimator at Feature Mean of {0} with {1} Features\n"
                 "for Column '{2}'")
        plt.title(title.format(feat_score, features.shape[0], yname), pad=10)
        plt.tight_layout()
        min_score_format = int(feat_score*100)
        plt.savefig(out_dir+"/figs/rfr_accuracy_{0}_{1}_new.png".format(yname,min_score_format))
        plt.close()

def add_features(data, id_name):

    #helper method to lookup correct postgres datatype
    def dtype_lookup(column_name):
        pd_type = data[column_name].dtype.name
        dict_dtype = {"object":"text",
                      "float64": "float",
                      "int64": "integer"}
        return dict_dtype[pd_type]
    with engine.begin() as cnx:
        for col in [c for c in data.columns if c != id_name]:
            #add column to `all_features` if it doesn't already exist
            data_type = dtype_lookup(col)
            sql_create = ("alter table all_features " 
                            "add column if not exists {0} {1}"
                            )
            cnx.execute(sql_create.format(col, data_type))
            #update current column in all_features with new values
            update_vals = {"column": col, "id_name": id_name}
            val_id = list(zip(data[col], data[id_name]))
            update_vals["values"] = ",".join("({0}, '{1}')".format(*i) 
                                        for i in val_id)
            sql = ("update all_features a set {column} =  n.{column} "
                    "from (values {values}) as n ({column}, {id_name}) "
                    "where n.{id_name} = a.{id_name}"
                    )
            cnx.execute(sql.format(**update_vals))

def more_parcels():

    sql = ("with par as ( "
            "select wkb_geometry, parcelid, st_area(p.wkb_geometry)/43560 par_acre, "  
                "sfcom, sfla, rmtot, res_par_perim, livunit "  
            "from built_env.sca_shelby_parcels_2017 p "
            "full join (select parid, sum(sf) sfcom from "  
                "built_env.sca_comintext group by parid) c "
                "on c.parid = parcelid "  
            "full join (select parid, sum(sfla) sfla, sum(rmtot) rmtot "  
                "from built_env.sca_dweldat group by parid) d "
                "on d.parid = parcelid "
            "full join "
            "(select parid, livunit, "
                "case when luc = '062' then st_perimeter(wkb_geometry) "
                    "else 0 end res_par_perim "
                "from built_env.sca_pardat, built_env.sca_shelby_parcels_2017 "
                "where parcelid = parid) pa "
            "on pa.parid = parcelid) "
            "select geoid10, par_acre, sfcom, sfla, rmtot, res_par_perim, livunit "
            "from par, tiger_tract_2010 t "
            "where st_intersects(st_centroid(par.wkb_geometry), t.wkb_geometry) "
        )
    df = pd.read_sql(sql, engine)
    df.fillna(0, inlace=True)
    grp = df.groupby("geoid10").median()
    grp.reset_index(inplace=True)
    add_features(grp, "geoid10")

def simpson_diversity():
    """Calculates a diversity index using Simpson's diversity index as represented
    by the formula:

        D = 1 - (sum(n(n-1))/N(N-1))
    where n is a total for a particular land use and N is the total land uses for
    a given geography.
    """
    sql = ("select luc, count(luc) ct_luc, geoid10 from built_env.sca_pardat, "
            "built_env.sca_shelby_parcels_2017 p, tiger_tract_2010 t "
            "where parid = parcelid and st_intersects(st_centroid(p.wkb_geometry), "
            "t.wkb_geometry) "
            "group by luc, geoid10"
            )

    df = pd.read_sql(sql, engine)
    #calculate n
    sp_count = lambda n: n * (n -1)
    df["ind_count"] = df.ct_luc.apply(sp_count)
    #calculate the diversity score, D, for all geographies
    def diversity_index(ind_count, all_count):
        return 1 - (sum(ind_count) / (sum(all_count)*(sum(all_count)-1)))
    div_score = pd.DataFrame(df.groupby("geoid10").apply(
                    lambda x: diversity_index(x["ind_count"], x["ct_luc"])))
    div_score.reset_index(inplace=True)
    div_score.rename(columns={0:"div_idx"}, inplace=True)
    add_features(div_score, "geoid10")

def parse_raster():
    """Steps taken to create data:
        1. Raster representation of land uses was created using gdal_rasterize in 
            QGIS using a cell size of 30 map units (feet). 
        2. Tract table was split into individual shapefiles based on geoid using
            Split vector layer tool in QGIS
        3. Split census tract shapefiles were used to split Raster created in 
            step 1 into individual rasters using gdal_wrap tool within shell 
            script
    TODO: 
        - automate rasterization using gdal_rasterize with postgresql layer
    
    """
    import pylandstats as pls
    
    raster_dir = "/home/natron/temp/split_raster"

    land_metrics = ["number_of_patches", "patch_density", "largest_patch_index",
                    "total_edge", "edge_density", "landscape_shape_index",
                    "contagion", "shannon_diversity_index"]
    all_geoids = []
    for img in os.listdir(raster_dir):
        geoid = img[8:-4]
        land = pls.Landscape(os.path.join(raster_dir, img))
        land_stats = land.compute_landscape_metrics_df()
        ls_dict = land_stats[land_metrics].to_dict("records")[0]
        ls_dict["geoid10"] = geoid
        all_geoids.append(ls_dict)
    df = pd.DataFrame(all_geoids)
    add_features(df, "geoid10")

def low_dimensional_plot():
    y = df["net"]
    x = df[x_vars]
    perplex = 5
    nrows, ncols = 5,5
    idx = 1
    for row in nrows:
        for col in ncols:
            x_fit = TSNE(n_components=2, perplexity=perplex).fit_transform(x)
            plt.subplot(nrows,ncols, idx, sharex=True, sharey=True)
            plt.scatter(x_fit[:,0], x[:,1], c=y, cmap=plt.get_cmap("PRGn"))
            plt.title("Perplexity: {}".format(perplex))
            perplex += 5
            idx += 1

def compare_city_owned():
    """
    By permit name:
    0.03% city-led construction
    96.5% city-led demolition
    By city-owned property
    0.6 % difference in construction
    0.2% difference in demolition 
    """
    q = ("select * "
          "from (select geoid10, "
          "count(distinct case when const_type = 'new' then permit end) numpermit,"
	      "count(distinct case when const_type = 'demo' then permit end) numdemo "
          "from permits p, tiger_tract_2010 t "
          "where st_within(p.wkb_geometry, t.wkb_geometry) {}"
          "group by t.geoid10) q "
        "order by geoid10"
    )

    omit = " and lower(p.name) not similar to '%(city of|cizty of)%' "
    df_all = pd.read_sql(text(q.format("")), engine)
    df_limit = pd.read_sql(text(q.format(omit)), engine)
    q_zoning = ("with p as (select wkb_geometry, "
		        "lower(regexp_replace(zoning, '[^a-zA-Z0-9]', '', 'g')) zoning "
		        "from built_env.sca_shelby_parcels_2017, built_env.sca_pardat "
		        "where parcelid = parid) "
                "select geoid10, zoning, "
                "count(zoning) from tiger_tract_2010 t, p "
                "where st_intersects(st_centroid(p.wkb_geometry), t.wkb_geometry) "
                "group by geoid10, zoning"                
                )
    dfz = pd.read_sql(text(q_zoning), engine)
    dfz_pivot = dfz.pivot(index="geoid10", columns="zoning", values="count")
    dfz_pivot.fillna(0., inplace=True)
    dfz_pivot.reindex(inplace=True)
    df = df.join(dfz_pivot, on="geoid10", how="left")
    q_cityown = ("with parcels as (select wkb_geometry "
	                "from sca_owndat, built_env.sca_shelby_parcels_2017 pa "
                    "where lower(concat(adrno,adrdir,adrstr,adrsuf)) <> '125nmainst' "
                    "and parcelid=parid) "
                  "select geoid10, "
                    "count(distinct case when const_type = 'new' then permit end) numpermit, "
                    "count(distinct case when const_type = 'demo' then permit end) numdemo "
                  "from permits p, tiger_tract_2010 t, parcels "
                    "where st_intersects(p.wkb_geometry, parcels.wkb_geometry) "
                    "and st_intersects(p.wkb_geometry, t.wkb_geometry) "
                    "group by geoid10"
                )
    df_cityown = pd.read_sql(q_cityown, engine)
#    chars = "[/\/#/$/*///-/(/)/s]"
#    dfz.zoning = dfz.zoning.str.replace(chars, "").str.lower()    

                

if __name__=="__main__":
    main()
