# Databricks notebook source
# MAGIC %run ./aruba_utility

# COMMAND ----------

spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
spark.conf.set("spark.sql.adaptive.autoBroadcastJoinThreshold", -1)

spark.conf.set("spark.sql.shuffle.partitions", 500)

spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "256MB")

# COMMAND ----------

from pyspark.sql.functions import col, broadcast

def transform_fn(cse_df):

    cse_col=["ce_cse_i_2_nm",
        "ce_cs_nr",
        "ce_opendate_dt",
        "ce_cse_aging_cd",
        "ce_cls_2_dt",
        "ce_rg_29_nm",
        "ce_crt_37_dt",
        "ce_last_mfy_39_dt",
        "ce_stt_130_nm",
        "ce_cse_rs_2_nm",
        "ce_svrty_nm",
        "ce_hgst_svrty",
        "ce_elevated_cd",
        "ce_esctd_cd",
        "ce_cls_cse_rsn_nm",
        "ce_initial_severity",
        "ce_sltn_dt_tm",
        "ce_cse_org_2_nm",
        "ce_rsl_2_nm",
        "ce_sftwr_vrsn",
        "ce_csr_prt_ord_cnt_cd",
        "ce_opn_onsite_tsk_cnt_cd",
        "ce_ctr_56_nm",
        "ce_cse_own_mgr_nm",
        "ce_dfct_id",
        "ce_prnt_cse_id_nm",
        "ce_prod_nr",
        "ce_prdct_grp",
        "ce_prod_l_3_nm",
        "ce_acct_i_4_nm",
        "ce_cs_nm",
        "ce_cse_sb_ctgy",
        "ce_acct_ltn_nm",
        "ce_cse_own_eml_nm",
        "ce_cntrct_i_nm",
        "ce_srv_ptfl_nm",
        "ce_case_cse_cgy_aruba__1_cd",
        "ce_cse_cgy_nm",
        "ce_re_2_nm",
        "ce_cse_grp_nm",
        "ce_rsln_cd",
        "ce_accountname_nm",
        "ce_acct_typ_nm",
        "ce_rsln_typ_nm",
        "ce_rsln_subcode_cd",
        "ce_elvtn_ind_nm",
        "ce_ato_cls_cd",
        "ce_prv_own_name",
        "ce_team_lead_eml_nm",
        "ce_cse_clsd_rsn",
        "ce_cse_opn_rsn",
        "ce_detailed_stts_c_cd",
        "ce_enttlmt_smmry_nm",
        "ce_prv_own_nm",
        "ce_sb_7_nm",
        "ce_cntct_i_9_nm",
        "ce_rsln_sbcgy_nm",
        "ce_rsln_subcode_cd",
        "ce_pr_12_nm",
        "ce_ever_esctd_cd",
        "ce_esctn_id_nm",
        "ce_esctn_team_nm",
        "ce_arb_cntrl_mngd",
        "ce_own_i_33_nm",
        "ce_asst_id_nm",
        "ce_asst_lctn_nm",
        "ce_ins_gmt_ts"]

    ac_cols = ["acct_id", 
               "acct_nm",
                "rgn_nm", 
                "prnt_acct_id", 
                "prnt_acct",
               "subregion1_nm",
                "countrycode_nm",
                 "subregion2_nm",
                  "subregion3_nm",
               "acct_rgn_nm",
                "emdm_prty_id",
                 "gbl_ent_id", 
                 "ctry_ent_id",
               "gbl_ent_nm", 
               "ctry_ent_nm",
                "acct_ltn_cptr_nm", 
                "mdcp_site_insn_id",
               "mdcp_site_id",
                "acct_st_id", 
                "acct_st_nm", 
                "top_prnt_st_id", 
                "top_prnt_st_nm"]

    us_cols = ["usr_id", 
               "emp_nr", 
               "fll_nm_1", 
               "last_nm", 
               "frst_nm", 
               "ctry_nm"]

    asst_cols = ["asst_id", 
                 "srl_nr", 
                 "asst_nm", 
                 "prod_nr"]
    lctn_cols = ["rec_id","lctr_id"]

    ce = cse_df
    ac = spark.table("nnbu.sch_nnbu_stage_vw.vw_ea_common_acct_dmnsn").select(*ac_cols)
    us = spark.table("nnbu.sch_nnbu_stage_vw.vw_sfdc_ea_common_usr_dmnsn").select(*us_cols)
    asst = spark.table("nnbu.sch_nnbu_stage_vw.vw_ea_common_asst_dmnsn").select(*asst_cols)
    lctn = spark.table("nnbu.sch_nnbu_stage_vw.vw_ea_common_lctn_dmnsn").select(*lctn_cols)

    # Prefix rename
    ce = ce.select([col(c).alias(f"ce_{c}") for c in ce.columns]).select(*cse_col)
    ac = ac.select([col(c).alias(f"ac_{c}") for c in ac.columns])
    us = us.select([col(c).alias(f"us_{c}") for c in us.columns])
    asst = asst.select([col(c).alias(f"asst_{c}") for c in asst.columns])
    ln = lctn.select([col(c).alias(f"ln_{c}") for c in lctn.columns])
    
    df = ce \
    .join((ac), col("ce_acct_i_4_nm") == col("ac_acct_id"), "left") \
    .join((us), col("ce_own_i_33_nm") == col("us_usr_id"), "left") \
    .join((asst), col("ce_asst_id_nm") == col("asst_asst_id"), "left") \
    .join((ln), col("ce_asst_lctn_nm") == col("ln_rec_id"), "left")

    
    selected_columns = [
        "ce_cse_i_2_nm",
        "ce_cs_nr",
        "ce_opendate_dt",
        "ce_cse_aging_cd",
        "ce_cls_2_dt",
        "ce_rg_29_nm",
        "ce_crt_37_dt",
        "ce_last_mfy_39_dt",
        "ce_stt_130_nm",
        "ce_cse_rs_2_nm",
        "ce_svrty_nm",
        "ce_hgst_svrty",
        "ce_elevated_cd",
        "ce_esctd_cd",
        "ce_cls_cse_rsn_nm",
        "ce_initial_severity",
        "ce_sltn_dt_tm",
        "ce_cse_org_2_nm",
        "ce_rsl_2_nm",
        "ce_sftwr_vrsn",
        "ce_csr_prt_ord_cnt_cd",
        "ce_opn_onsite_tsk_cnt_cd",
        "ce_ctr_56_nm",
        "ce_cse_own_mgr_nm",
        "ce_dfct_id",
        "ce_prnt_cse_id_nm",
        "ce_prod_nr",
        "ce_prdct_grp",
        "ce_prod_l_3_nm",
        "ce_acct_i_4_nm",
        "ce_cs_nm",
        "ce_cse_sb_ctgy",
        "ce_acct_ltn_nm",
        "ce_cse_own_eml_nm",
        "ce_cntrct_i_nm",
        "ce_srv_ptfl_nm",
        "ce_case_cse_cgy_aruba__1_cd",
        "ce_cse_cgy_nm",
        "ce_re_2_nm",
        "ce_cse_grp_nm",
        "ce_rsln_cd",
        "ce_accountname_nm",
        "ce_acct_typ_nm",
        "ce_rsln_typ_nm",
        "ce_rsln_subcode_cd",
        "ce_elvtn_ind_nm",
        "ce_ato_cls_cd",
        "ce_prv_own_name",
        "ce_team_lead_eml_nm",
        "ce_cse_clsd_rsn",
        "ce_cse_opn_rsn",
        "ce_detailed_stts_c_cd",
        "ce_enttlmt_smmry_nm",
        "ce_prv_own_nm",
        "ce_sb_7_nm",
        "ce_cntct_i_9_nm",
        "ce_rsln_sbcgy_nm",
        "ce_pr_12_nm",
        "ce_ever_esctd_cd",
        "ce_esctn_id_nm",
        "ce_esctn_team_nm",
        "ce_arb_cntrl_mngd",
        "ce_own_i_33_nm",
        "ce_asst_id_nm",
        "ce_asst_lctn_nm",
        "us_usr_id",
        "us_emp_nr",
        "us_fll_nm_1",
        "us_last_nm",
        "us_frst_nm",
        "us_ctry_nm",
        "ac_acct_id",
        "ac_acct_nm",
        "ac_rgn_nm",
        "ac_prnt_acct_id",
        "ac_prnt_acct",
        "ac_subregion1_nm",
        "ac_countrycode_nm",
        "ac_subregion2_nm",
        "ac_subregion3_nm",
        "ac_acct_rgn_nm",
        "ac_emdm_prty_id",
        "ac_gbl_ent_id",
        "ac_ctry_ent_id",
        "ac_gbl_ent_nm",
        "ac_ctry_ent_nm",
        "ac_acct_ltn_cptr_nm",
        "ac_mdcp_site_insn_id",
        "ac_mdcp_site_id",
        "ac_acct_st_id",
        "ac_acct_st_nm",
        "ac_top_prnt_st_id",
        "ac_top_prnt_st_nm",
        "asst_asst_id",
        "asst_srl_nr",
        "asst_asst_nm",
        "asst_prod_nr",
        "ln_lctr_id",
        "ln_rec_id",
        "ce_ins_gmt_ts"
    ]

    df_final = df.select(*selected_columns)
    df_final=df_final.withColumn("ins_dtm", current_timestamp())\
                     .withColumn("updt_dtm", current_timestamp())    
    return df_final

# COMMAND ----------

import traceback
from pyspark.sql.functions import *
from pyspark import StorageLevel

pipeline_name = "pl_de_trigger_dbx_aruba_case"
driver_table = "nnbu.sch_nnbu_stage_vw.vw_ea_common_cse_dmnsn"
retry_table = "nnbu.nnbu_cases.aruba_case_consolidate_retry"
control_table = "nnbu.nnbu_cases.etl_incremental_control"
target_table="nnbu.nnbu_cases.aruba_cases"
driver_pk = "cse_i_2_nm"
driver_watermark_col = "ins_gmt_ts"



# COMMAND ----------

try:
    watermark = get_watermark(pipeline_name, driver_table, control_table)
    print(f"Last Processed {driver_watermark_col} values : {watermark}")

    driver_df = spark.table(driver_table)
    incremental_df = read_incremental(driver_table, driver_watermark_col, watermark)
    retry_df = read_retry(driver_df, incremental_df, retry_table, driver_pk)

    final_input = incremental_df.unionByName(
        retry_df,
        allowMissingColumns=True
    )

    if final_input.isEmpty():
        print("No data to process")
        dbutils.notebook.exit("No data to process")

    try:
        derived_aruba_case_df = transform_fn(final_input)
        derived_aruba_case_df = target_schema_conversion(derived_aruba_case_df, target_table)
        derived_aruba_case_df.createOrReplaceTempView("final_view")
    except:
        raise Exception("Transformation failed\n" + traceback.format_exc())
    success_condition = (
       # col("ce_acct_i_4_nm").isNotNull() &
        col("ac_acct_id").isNotNull() &
       #col("ce_own_i_33_nm").isNotNull() &
        col("us_usr_id").isNotNull() &
       # col("ce_asst_id_nm").isNotNull() &
        col("asst_asst_id").isNotNull() 
    )
    
    try:
        if not derived_aruba_case_df.isEmpty():
            spark.sql(f"""
                DELETE FROM {target_table} AS t
                WHERE EXISTS (
                    SELECT 1
                    FROM final_view AS d
                    WHERE t.ce_cse_i_2_nm = d.ce_cse_i_2_nm
                )
            """)
            derived_aruba_case_df.write.format("delta").option("mergeSchema", "True").mode("append").saveAsTable(target_table)
            print(f"Aruba Case Data process completed into the table {target_table}")
        else:
            print("No records to write")
    except Exception as e:
        print("Exception occurred during Aruba Case Data processing:")
        print(str(e))

    #spark.sql(f"""
    #MERGE INTO {target_table} t
    #USING final_view s
    #ON t.ce_cse_i_2_nm = s.ce_cse_i_2_nm
    #WHEN MATCHED THEN UPDATE SET *
    #  EXCEPT (create_dtm)
    #WHEN NOT MATCHED THEN INSERT *
    #""")
    print(f"Data Load is complted to the table : {target_table}")

    success_df = derived_aruba_case_df.filter(success_condition)
    failed_keys_df = derived_aruba_case_df.filter(~success_condition).select(col("ce_cse_i_2_nm").alias(driver_pk)).distinct()
    
    upsert_retry_tracker(failed_keys_df, retry_table, driver_pk) 

    success_df.createOrReplaceTempView("success_view")
    spark.sql(f"""
    DELETE FROM {retry_table} r
    WHERE EXISTS (
        SELECT 1
        FROM success_view s
        WHERE r.primary_key = s.ce_cse_i_2_nm
    )
    """)
   
    update_watermark(
       incremental_df,
       control_table,
       pipeline_name,
       driver_table,
       driver_watermark_col
    )
    
    print(f"{pipeline_name} COMPLETED")
except Exception as e:
    print("PIPELINE FAILED")
    print(traceback.format_exc())
    raise e