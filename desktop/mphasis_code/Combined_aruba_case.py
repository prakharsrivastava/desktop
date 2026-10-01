# Databricks notebook source
# MAGIC %run ./aruba_utility

# COMMAND ----------

from pyspark.sql.functions import collect_list, col, struct, current_timestamp, lit, when, to_date
import json

def target_schema_conversion(input_df, catalog_table_name):
    target_catalog_df = spark.table(catalog_table_name)
    # Create lowercase mapping of target schema
    target_schema = {field.name.lower(): field.dataType for field in target_catalog_df.schema.fields}
    # Apply casting by matching lowercase column names
    input_df = input_df.select([
        col(c).cast(target_schema[c.lower()]).alias(c) if c.lower() in target_schema else col(c)
        for c in input_df.columns])
    target_columns = target_catalog_df.columns
    input_order_df = input_df.select([col(c).alias(c.lower()) for c in target_columns])
    return input_order_df


# COMMAND ----------

# DBTITLE 1,Cell 3
def transform_fn(aruba_cases_incr):
    aruba_cases_incr.createOrReplaceTempView("aruba_cases_incr")
    aruba_case_df = spark.sql(f"""
            SELECT 
        'HPE' as source,
        ce.ce_cse_i_2_nm AS case_id,
        ce.ce_cs_nr AS case_number,
        NULL AS edp_created_date,
        NULL AS edp_lastmodified_date,
        ce.ce_opendate_dt AS case_date_created,
        cal.cldr_trns_dt AS case_date_closed_date,
        ce.ce_cse_aging_cd AS case_age_days,
        ce.ce_cls_2_dt AS case_date_closed_date_timestamp,
        NULL AS end_customer_global_ultimate_parent_id,
        NULL AS end_customer_global_ultimate_parent_name_text,
        NULL AS end_customer_name_text,
        NULL AS end_customer_parent_id,
        NULL AS end_customer_parent_name_text,
        ce.ce_rg_29_nm AS end_customer_theater,
        NULL AS end_customer_services_theater,
        NULL AS end_customer_ultimate_parent_id,
        NULL AS end_customer_ultimate_parent_name_text,
        NULL AS product_family,
        NULL AS customer_ultimate_parent_id,
        ce.ce_crt_37_dt AS case_date_created_timestamp,
        NULL AS customer_ultimate_parent_name_text,
        NULL AS case_owner_employee_company_name,
        NULL AS created_by_employee_employee_name,
        NULL AS case_date_perm_fix_timestamp,
        ce.ce_last_mfy_39_dt AS case_last_modified_date_timestamp,
        ce.ce_stt_130_nm AS case_status,
        NULL AS case_cat_cause2,
        NULL AS case_cat_cause1,
        NULL AS case_cat_cause3,
        CONCAT(LEFT(cal.cldr_yr_mth_cd, 4), '-M', RIGHT(cal.cldr_yr_mth_cd, 2))  AS case_date_closed_month,
        NULL AS tech_cat_category_1,
        NULL AS tech_cat_category_2,
        NULL AS tech_cat_category_3,
        ce.ce_cse_rs_2_nm AS reason,
        REPLACE(cal.cldr_yr_qtr_cd, 'FY', '') AS case_date_closed_qtr,
        CONCAT(LEFT(cal.cldr_yr_wk_nr, 4), '-W', RIGHT(cal.cldr_yr_wk_nr, 2)) AS case_date_closed_week,
        ce.ce_svrty_nm AS priority_current,
        cal.wk_d_num AS case_date_closed_weekday,
        ce.ce_hgst_svrty AS priority_highest,
        CASE when ce.ce_elevated_cd=0 THEN 'FALSE' 
        ELSE 'TRUE' 
        END AS flag_elevated,
        CASE when ce.ce_esctd_cd=0 THEN 'FALSE' 
        ELSE 'TRUE' 
        END AS flag_etc,
        Replace(cal.fisc_yr_qrtr_cd, 'FY', '') AS case_date_closed_hpe_qtr,
        EXTRACT(YEAR FROM ce.ce_cls_2_dt) AS case_date_closed_year,
        REPLACE(cal_op.cldr_yr_qtr_cd, 'FY', '') AS case_date_created_qtr,
        CONCAT(LEFT(cal_op.cldr_yr_wk_nr, 4), '-W', RIGHT(cal_op.cldr_yr_wk_nr, 2)) AS case_date_created_week,
        ce.ce_cls_cse_rsn_nm AS closed_reason,
        cal_op.wk_d_num AS case_date_created_weekday,
        EXTRACT(YEAR FROM ce.ce_opendate_dt) AS case_date_created_year,
        NULL AS case_contract_service_sku,
        NULL AS case_contract_type,
        ce.asst_srl_nr AS entitled_serial_number,
        NULL AS case_cat_case_type,
        CASE 
        WHEN 
            DATEDIFF(
                second,
                ce.ce_crt_37_dt,
                COALESCE(
                    LEAST(ce.ce_sltn_dt_tm, ce.ce_cls_2_dt),
                    ce.ce_sltn_dt_tm,
                    ce.ce_cls_2_dt
                )
            ) <= 86400   
        THEN 'TRUE'
        ELSE 'FALSE'
        END AS flag_first_day_resolution,
        NULL AS case_owner_username,
        ce.ce_initial_severity AS priority_initial,
        NULL AS product_category,
        NULL AS initial_response_time,
        CASE
            WHEN ce.ce_cls_2_dt IS NULL THEN NULL
            WHEN EXTRACT(MINUTE FROM ce.ce_cls_2_dt) BETWEEN 0 AND 30 THEN '0-30'
            ELSE '30-60'
        END AS case_date_closed_1_2_hr_interval,
        CASE
            WHEN ce.ce_cls_2_dt IS NULL THEN NULL
            WHEN EXTRACT(MINUTE FROM ce.ce_cls_2_dt) BETWEEN 0 AND 15 THEN '0-15'
            WHEN EXTRACT(MINUTE FROM ce.ce_cls_2_dt) BETWEEN 16 AND 30 THEN '15-30'
            WHEN EXTRACT(MINUTE FROM ce.ce_cls_2_dt) BETWEEN 31 AND 45 THEN '30-45'
            WHEN EXTRACT(MINUTE FROM ce.ce_cls_2_dt) BETWEEN 46 AND 59 THEN '45-60'
        END AS case_date_closed_1_4_hr_interval,
        EXTRACT(HOUR FROM ce.ce_cls_2_dt) AS case_date_closed_hour,
        NULL AS case_category,
        CASE
            WHEN ce.ce_crt_37_dt IS NULL THEN NULL
            WHEN EXTRACT(MINUTE FROM ce.ce_crt_37_dt) BETWEEN 0 AND 30 THEN '0-30'
            ELSE '30-60'
        END AS case_date_created_1_2_hr_interval,
        CASE
            WHEN ce.ce_crt_37_dt IS NULL THEN NULL
            WHEN EXTRACT(MINUTE FROM ce.ce_crt_37_dt) BETWEEN 0 AND 15 THEN '0-15'
            WHEN EXTRACT(MINUTE FROM ce.ce_crt_37_dt) BETWEEN 16 AND 30 THEN '15-30'
            WHEN EXTRACT(MINUTE FROM ce.ce_crt_37_dt) BETWEEN 31 AND 45 THEN '30-45'
            WHEN EXTRACT(MINUTE FROM ce.ce_crt_37_dt) BETWEEN 46 AND 59 THEN '45-60'
        END AS case_date_created_1_4_hr_interval,
        NULL AS case_record_type,
        EXTRACT(HOUR FROM ce.ce_crt_37_dt) AS case_date_created_hour,
    CONCAT(LEFT(cal_op.cldr_yr_mth_cd, 4), '-M', RIGHT(cal_op.cldr_yr_mth_cd, 2)) AS case_date_created_month,
        ce.ce_cse_org_2_nm AS case_source,
        ce.ce_rsl_2_nm AS case_resolution_notes,
        ce.us_emp_nr AS case_owner_employee_id,
        NULL AS platform,
        NULL AS case_cat_case_subtype,
        datediff(from_utc_timestamp(current_timestamp(), 'America/Los_Angeles'),ce.ce_last_mfy_39_dt) AS case_inactive_days,
        NULL AS product_series,
        ce.ce_sftwr_vrsn AS sw_release,
        NULL AS case_routing_type,
        CASE
        WHEN ce.ce_csr_prt_ord_cnt_cd >= 0
        OR ce.ce_opn_onsite_tsk_cnt_cd >= 0
        THEN 'TRUE'
        ELSE 'FALSE'
        END AS flag_rma_attached,
        NULL AS product_line,
        ce.ce_ctr_56_nm AS customer_country,
        NULL AS customer_id,
        NULL AS customer_name_text,
        NULL AS customer_parent_id,
        NULL AS customer_parent_name_text,
        ce.ce_rg_29_nm AS customer_theater,
        NULL AS case_owner_employee_country,
        NULL AS case_owner_employee_function,
        NULL AS case_owner_employee_product,
        ce.us_fll_nm_1 AS case_owner_employee_name,
        NULL AS case_owner_employee_role,
        NULL AS case_owner_l4_director,
        NULL AS case_owner_l5,
        NULL AS case_owner_l6,
        NULL AS case_owner_outsourcer,
        NULL AS created_by_employee_employee_function,
        ce.ce_cse_own_mgr_nm AS case_owner_supervisor,
        NULL AS case_owner_supervisor_username,
        ce.ce_dfct_id AS first_linked_pr_number,
        CASE
        WHEN ce.ce_dfct_id IS NOT NULL THEN 'FALSE'
        ELSE 'TRUE'
        END  AS flag_mttcxe,
        CASE
        WHEN ce.ce_dfct_id IS NOT NULL THEN 'TRUE'
        ELSE 'FALSE'
        END AS flag_mtte,
        CASE
        WHEN ce.ce_dfct_id IS NOT NULL THEN 'TRUE'
        ELSE 'FALSE'
        END AS flag_pr_currently_linked,
        CASE 
        WHEN ce.ce_dfct_id IS NULL 
            THEN CAST(EXTRACT(DAY FROM (ce.ce_sltn_dt_tm - ce.ce_crt_37_dt)) + 
        EXTRACT(HOUR FROM (ce.ce_sltn_dt_tm - ce.ce_crt_37_dt)) / 24.0 + 
        EXTRACT(MINUTE FROM (ce.ce_sltn_dt_tm - ce.ce_crt_37_dt)) / 1440.0 
        AS DECIMAL(9, 2))
        END AS mttcxe_mean_time_to_customer_without_engineering_escalation,
        ce.ce_dfct_id AS primary_pr_number,
        CASE 
        WHEN ce.ce_sltn_dt_tm IS NOT NULL THEN 'TRUE'
        ELSE 'FALSE'
        END  AS flag_permfix_case,
        ce.ce_csr_prt_ord_cnt_cd AS csr_part_order_count,
        ce.ce_prnt_cse_id_nm AS parent_case_number_id,
        ce.ce_opn_onsite_tsk_cnt_cd AS open_onsite_task_count,
        CAST(SUBSTRING(cal.fisc_yr_mnth_cd, 8, 2) AS INT) AS case_closed_hpe_month_number,
        CAST(SUBSTRING(cal_op.fisc_yr_mnth_cd, 8, 2) AS INT)  AS case_created_hpe_month_number,
        CONCAT(cal.drvd_fisc_yr_nm, '-W',cal.nnbu_fscl_wk_nr)  AS case_closed_hpe_week_number,
        CONCAT(cal_op.drvd_fisc_yr_nm, '-W',cal_op.nnbu_fscl_wk_nr) AS case_created_hpe_week_number,
        REPLACE(cal.fisc_yr_qrtr_cd,'FY','') AS case_closed_hpe_quarter,
        REPLACE(cal_op.fisc_yr_qrtr_cd,'FY','')  AS case_created_hpe_quarter,
        cal.drvd_fisc_yr_nm AS case_closed_hpe_year,
        cal_op.drvd_fisc_yr_nm AS case_created_hpe_year,
        NULL AS case_type,
        ce.ce_prod_nr AS aruba_product_number,
        ce.ce_prdct_grp AS aruba_product_group_engg,
        ce.ce_prod_l_3_nm AS aruba_product_line,
        ce.asst_asst_nm AS aruba_asset_name,
        ce.ce_acct_i_4_nm AS aruba_account_id,
        ce.ac_acct_nm AS aruba_account_name,
        ce.ac_rgn_nm AS aruba_case_region_name,
        NULL AS distributor_party_id,
        NULL AS reseller_party_id,
        NULL AS customer_party_id,
        NULL AS hpe_business_area_cd,
        ce.ce_cs_nm AS aruba_case_cause_category,
        ce.ce_cse_sb_ctgy AS aruba_case_sub_category,
        ce.ce_acct_i_4_nm AS aruba_case_account_id,
        ce.ce_acct_ltn_nm AS aruba_case_account_latin_name,
        ce.ac_prnt_acct_id AS aruba_case_parent_account_id,
        ce.ac_prnt_acct AS aruba_case_parent_account_name,
        ce.ce_cse_own_eml_nm AS aruba_case_owner_email,
        ce.ac_subregion1_nm AS aruba_case_sub_region_name,
        ce.ac_countrycode_nm AS aruba_case_country_code,
        ce.ac_subregion2_nm AS aruba_case_sub_region_2_name,
        ce.ac_subregion3_nm AS aruba_case_sub_region_3_name,
        ce.ac_acct_rgn_nm AS aruba_case_account_region_name,
        ce.ac_emdm_prty_id AS aruba_case_party_id,
        ce.ac_gbl_ent_id AS aruba_case_ge_id,
        ce.ac_ctry_ent_id AS aruba_case_ce_id,
        ce.ac_gbl_ent_nm AS aruba_case_ge_name,
        ce.ac_ctry_ent_nm AS aruba_case_ce_name,
        ce.us_last_nm AS aruba_case_owner_last_name,
        ce.us_frst_nm AS aruba_case_owner_first_name,
        ce.us_ctry_nm AS aruba_case_owner_country,
        ce.asst_prod_nr AS aruba_case_product_number,
        ce.ce_cntrct_i_nm AS aruba_case_contract_identifier,
        ce.ce_srv_ptfl_nm AS aruba_case_service_portfolio,
        ce.ce_case_cse_cgy_aruba__1_cd AS aruba_case_category_cd,
        ce.ce_cse_cgy_nm AS aruba_case_category_name,
        ce.ce_re_2_nm AS aruba_case_record_type,
        NULL as initial_case_owner_name,
        NULL as case_first_assigned_initial_owner_timestamp,
        current_timestamp() as ins_dtm,
        current_timestamp() as updt_dtm
        , ce.ce_cse_grp_nm as aruba_case_group
        , ce.ce_rsln_cd as aruba_resolution_code
        , ce.ce_rg_29_nm as aruba_case_sfdc_region_name
        , case WHEN ce.ce_prnt_cse_id_nm != '0' 
                AND (ce.ce_csr_prt_ord_cnt_cd>= 0 OR ce.ce_csr_prt_ord_cnt_cd >= 0) 
                THEN 'RMA Case'
            WHEN ce.ce_prnt_cse_id_nm != '0' 
                THEN 'Assistance Case'
            ELSE 'Non-RMA Case'
        END AS aruba_case_cat_case_type
        , date_format(to_timestamp(ce.ce_sltn_dt_tm),"yyyy-MM-dd'T'HH:mm:ss.SSS") as aruba_resolution_date_time
        , ce.ac_acct_ltn_cptr_nm as aruba_account_latin_name
        , ce.ce_accountname_nm as aruba_case_account_nm
        , ce.ce_acct_typ_nm as aruba_case_account_type_name
        , ce.ac_mdcp_site_insn_id as aruba_account_mdcp_site_instance_id
        , ce.ac_mdcp_site_id as aruba_account_mdcp_site_id
        , ce.ac_acct_st_id as aruba_account_st_id
        , ce.ac_acct_st_nm as aruba_account_st_name
        , ce.ac_top_prnt_st_id as aruba_account_top_parent_st_id
        , ce.ac_top_prnt_st_nm as aruba_account_top_parent_st_name
        , ce.ce_rsln_typ_nm as aruba_resolution_type
        , ce.ce_rsln_subcode_cd as aruba_resolution_sub_code
        , ce.ce_elvtn_ind_nm as aruba_elevation_indicator
        , ce.ce_ato_cls_cd as aruba_auto_close
        , ce.ln_lctr_id as aruba_asset_location
        , ce.ce_prv_own_name as aruba_previous_owner_name
        , ce.ce_team_lead_eml_nm as aruba_team_lead_email
        , ce.ce_cse_clsd_rsn as aruba_case_closed_reason
        , ce.ce_cse_opn_rsn as aruba_case_open_reason
        , ce.ce_detailed_stts_c_cd as aruba_detailed_status
        , ce.ce_enttlmt_smmry_nm as aruba_entitlement_summary
        , ce.ce_prv_own_nm as aruba_previous_owner_id
        , ce.ce_sb_7_nm as aruba_case_subject
        , ce.ce_cntct_i_9_nm as auba_contactid
        , ce.ce_rsln_sbcgy_nm as aruba_resolution_sub_category
        , ce.ce_rsln_subcode_cd as aruba_resoultion_sub_code
        , ce.ce_pr_12_nm as aruba_priority
        , ce.ce_ever_esctd_cd as aruba_ever_escalated
        , ce.ce_esctn_id_nm as aruba_escalation_id
        , ce.ce_esctn_team_nm as aruba_escalation_team_name
        , ce.ce_arb_cntrl_mngd as aruba_centrally_managed
        , COALESCE(concat(substr(cal_op.fisc_yr_mnth_cd, 3, 4), '-M', right(cal_op.fisc_yr_mnth_cd, 2)), NULL) AS case_created_hpe_month
        , COALESCE(concat(substr(cal.fisc_yr_mnth_cd, 3, 4), '-M', right(cal.fisc_yr_mnth_cd, 2)), NULL) AS  case_closed_hpe_month
            FROM 
                aruba_cases_incr ce
            LEFT JOIN nnbu.sch_nnbu_stage_vw.vw_nnbu_clndr_dmnsn cal
                ON cal.cldr_trns_dt = TO_DATE(ce.ce_cls_2_dt)
            LEFT JOIN nnbu.sch_nnbu_stage_vw.vw_nnbu_clndr_dmnsn cal_op
                ON cal_op.cldr_trns_dt = TO_DATE(ce.ce_opendate_dt)
        """)

    aruba_case_df.createOrReplaceTempView("aruba_case_temp_view")
        
    final_aruba_case_df = spark.sql(f"""
            SELECT  
        acv.*,
        p1.Product_Category AS aruba_product_category_derived_P1,
        p2.Product_Category AS aruba_product_category_derived_P2,
        p3.Product_Category AS aruba_product_category_derived_P3,
        COALESCE(p1.Product_Category, p2.Product_Category, p3.Product_Category) AS aruba_product_category_derived,
        acv.aruba_case_owner_email as case_owner_email,
        NULL as hw_sw_products,
        com.manager_email as case_owner_supervisor_email,
        com.primary_product_category as aruba_queue_primary_product_category,
        com.updated_queue as aruba_case_owner_updated_queue,
        CASE 
            WHEN com.updated_queue IN ('GEC', 'ERT', 'GSC', 'GSC-Aruba', 'ART') THEN 'TAC'
            ELSE 'Non TAC'
        END AS aruba_case_owner_updated_queue_group,
        com.updated_case_owner_manager as aruba_queue_supervisor_name,
        com.role as aruba_queue_case_owner_role,
        com.sub_queue as aruba_queue_case_owner_sub_queue,
        com.primary_product_group as aruba_queue_primary_product_group,
        com.org as aruba_queue_case_owner_org,
        com.l3 as aruba_queue_case_owner_L3,
        com.l4 as aruba_queue_case_owner_L4,
        com.l5 as aruba_queue_case_owner_L5,
        com.l6 as aruba_queue_case_owner_L6,
        com.l7 as aruba_queue_case_owner_L7,
        com.hpe_tower_manager as aruba_queue_hpe_tower_manager,
        com.premium_team as aruba_queue_premium_team,
        com.country as aruba_queue_case_owner_country,
        com.state_city as aruba_queue_case_owner_state_city,
        com.location as aruba_queue_case_owner_location,
        com.updated_primary_product_group as aruba_queue_updated_primary_product_group,
        com.updated_case_owner_manager as aruba_queue_case_owner_manager_name
    FROM aruba_case_temp_view acv
    LEFT JOIN (
        SELECT DISTINCT product_number_name, Product_Category
        FROM nnbu.sch_nnbu_rptng.vw_aruba_pc_mapping
        WHERE Priority = 'P1'
    ) p1
        ON p1.product_number_name = COALESCE(acv.aruba_case_product_number, acv.aruba_product_number)
    LEFT JOIN (
        SELECT DISTINCT SFDC_Product_Group, Product_Category
        FROM nnbu.sch_nnbu_rptng.vw_aruba_pc_mapping
        WHERE Priority = 'P2'
    ) p2
        ON p2.SFDC_Product_Group = acv.aruba_product_group_engg
    LEFT JOIN (
        SELECT DISTINCT product_number_name, Product_Category
        FROM nnbu.sch_nnbu_rptng.vw_aruba_pc_mapping
        WHERE Priority = 'P3'
    ) p3
        ON p3.product_number_name = COALESCE(acv.aruba_case_product_number, acv.aruba_product_number)
    LEFT JOIN nnbu.sch_nnbu_rptng.vw_aruba_case_owner_mapping com
        ON acv.aruba_case_owner_email = com.hpe_email_id
        AND (
            CASE 
                WHEN acv.case_date_closed_month IS NOT NULL THEN acv.case_date_closed_month
                ELSE acv.case_date_created_month
            END = com.monthyear
        )
        """)
    return final_aruba_case_df


# COMMAND ----------

import traceback
from pyspark.sql.functions import *
from pyspark import StorageLevel

# CONFIG
pipeline_name = "pl_de_trigger_dbx_aruba_combined_case"
driver_table = "nnbu.nnbu_cases.aruba_cases"
control_table = "nnbu.nnbu_cases.etl_incremental_control"
target_table="nnbu.nnbu_cases.jnpr_aruba_cases"
driver_pk = "ce_cse_i_2_nm"
driver_watermark_col = "ce_ins_gmt_ts"



# COMMAND ----------

import traceback
try:
    watermark = get_watermark(pipeline_name, driver_table, control_table)
    print(f"previous watermark value is {watermark}")
    driver_df = spark.table(driver_table)
    incremental_df = read_incremental(driver_table, driver_watermark_col, watermark)
    # retry_df = read_retry(driver_df, incremental_df, retry_table, driver_pk)
    # final_input = incremental_df.unionByName(
    #     retry_df,
    #    allowMissingColumns=True
    #)

    if incremental_df.limit(1).count() == 0:
        print("No data to process")
        dbutils.notebook.exit("No data to process")

    try:
        derived_aruba_case_df = transform_fn(incremental_df)
        derived_aruba_case_df = target_schema_conversion(derived_aruba_case_df, target_table)
        derived_aruba_case_df.createOrReplaceTempView("aruba_final_view")
    except:
        raise Exception("Transformation failed\n" + traceback.format_exc())

    try:
        if derived_aruba_case_df.first() is not None:
            spark.sql(f"""
                DELETE FROM {target_table} t
                WHERE t.source = 'HPE'
                AND EXISTS (
                    SELECT 1
                    FROM aruba_final_view v
                    WHERE t.case_id = v.case_id
                )
            """)
            derived_aruba_case_df.write.format("delta").option("mergeSchema", "True").mode("append").saveAsTable(target_table)
            print(f"Aruba Case Data process completed into the table {target_table}")
        else:
            print("No records to write")
    except Exception as e:
        print("Exception occurred during Aruba Case Data processing:")
        print(str(e))
    
    # Insert data from final view into target table
    #derived_aruba_case_df =derived_aruba_case_df.dropDuplicates(["case_id"])
    #derived_aruba_case_df.write.option("mergeSchema", "true").format("delta").mode("append").saveAsTable(target_table)
    

    #spark.sql(f"""
    #MERGE INTO {target_table} t
    #USING aruba_final_view s
    #ON t.case_id = s.case_id
    #WHEN MATCHED AND t.source == 'HPE' THEN UPDATE SET *
    #WHEN NOT MATCHED and s.source == 'HPE' THEN INSERT *
    #""")

   # failed_keys_df = derived_aruba_case_df.filter(~success_condition).select(col("case_id").alias(driver_pk))
    
    #upsert_retry_tracker(failed_keys_df, retry_table, driver_pk)

    #success_df.createOrReplaceTempView("success_view")
    #spark.sql(f"""
    #DELETE FROM {retry_table}
    #WHERE primary_key IN (SELECT case_id FROM success_view)
    #""")
    
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

# COMMAND ----------



# COMMAND ----------

case_owner_aruba_case_df = spark.sql("""
    SELECT
        acv.case_id,
        com.manager_email as case_owner_supervisor_email,
        com.primary_product_category as aruba_queue_primary_product_category,
        com.updated_queue as aruba_case_owner_updated_queue,
        CASE 
            WHEN com.updated_queue IN ('GEC', 'ERT', 'GSC', 'GSC-Aruba', 'ART') THEN 'TAC'
            ELSE 'Non TAC'
        END AS aruba_case_owner_updated_queue_group,
        com.updated_case_owner_manager as aruba_queue_supervisor_name,
        com.role as aruba_queue_case_owner_role,
        com.sub_queue as aruba_queue_case_owner_sub_queue,
        com.primary_product_group as aruba_queue_primary_product_group,
        com.org as aruba_queue_case_owner_org,
        com.l3 as aruba_queue_case_owner_L3,
        com.l4 as aruba_queue_case_owner_L4,
        com.l5 as aruba_queue_case_owner_L5,
        com.l6 as aruba_queue_case_owner_L6,
        com.l7 as aruba_queue_case_owner_L7,
        com.hpe_tower_manager as aruba_queue_hpe_tower_manager,
        com.premium_team as aruba_queue_premium_team,
        com.country as aruba_queue_case_owner_country,
        com.state_city as aruba_queue_case_owner_state_city,
        com.location as aruba_queue_case_owner_location,
        com.updated_primary_product_group as aruba_queue_updated_primary_product_group,
        com.updated_case_owner_manager as aruba_queue_case_owner_manager_name
    FROM nnbu.nnbu_cases.jnpr_aruba_cases acv
    INNER  JOIN nnbu.sch_nnbu_rptng.vw_aruba_case_owner_mapping com
        ON acv.aruba_case_owner_email = com.hpe_email_id
        AND (
            CASE 
                WHEN acv.case_date_closed_month IS NOT NULL THEN acv.case_date_closed_month
                ELSE acv.case_date_created_month
            END = com.monthyear
        )
    WHERE acv.source = 'HPE'
    AND (
    (acv.case_owner_supervisor_email != com.manager_email OR (acv.case_owner_supervisor_email IS NULL AND com.manager_email IS NOT NULL)) OR
    (acv.aruba_queue_supervisor_name != com.updated_case_owner_manager OR (acv.aruba_queue_supervisor_name IS NULL AND com.updated_case_owner_manager IS NOT NULL)) OR
    (acv.aruba_queue_primary_product_category != com.primary_product_category OR (acv.aruba_queue_primary_product_category IS NULL AND com.primary_product_category IS NOT NULL)) OR
    (acv.aruba_case_owner_updated_queue != com.updated_queue OR (acv.aruba_case_owner_updated_queue IS NULL AND com.updated_queue IS NOT NULL)) OR
    (acv.aruba_case_owner_updated_queue_group != CASE WHEN com.updated_queue IN ('GEC', 'ERT', 'GSC', 'GSC-Aruba', 'ART') THEN 'TAC' ELSE 'Non TAC' END OR (acv.aruba_case_owner_updated_queue_group IS NULL AND CASE WHEN com.updated_queue IN ('GEC', 'ERT', 'GSC', 'GSC-Aruba', 'ART') THEN 'TAC' ELSE 'Non TAC' END IS NOT NULL)) OR
    (acv.aruba_queue_case_owner_role != com.role OR (acv.aruba_queue_case_owner_role IS NULL AND com.role IS NOT NULL)) OR
    (acv.aruba_queue_case_owner_sub_queue != com.sub_queue OR (acv.aruba_queue_case_owner_sub_queue IS NULL AND com.sub_queue IS NOT NULL)) OR
    (acv.aruba_queue_primary_product_group != com.primary_product_group OR (acv.aruba_queue_primary_product_group IS NULL AND com.primary_product_group IS NOT NULL)) OR
    (acv.aruba_queue_case_owner_org != com.org OR (acv.aruba_queue_case_owner_org IS NULL AND com.org IS NOT NULL)) OR
    (acv.aruba_queue_case_owner_L3 != com.l3 OR (acv.aruba_queue_case_owner_L3 IS NULL AND com.l3 IS NOT NULL)) OR
    (acv.aruba_queue_case_owner_L4 != com.l4 OR (acv.aruba_queue_case_owner_L4 IS NULL AND com.l4 IS NOT NULL)) OR
    (acv.aruba_queue_case_owner_L5 != com.l5 OR (acv.aruba_queue_case_owner_L5 IS NULL AND com.l5 IS NOT NULL)) OR
    (acv.aruba_queue_case_owner_L6 != com.l6 OR (acv.aruba_queue_case_owner_L6 IS NULL AND com.l6 IS NOT NULL)) OR
    (acv.aruba_queue_case_owner_L7 != com.l7 OR (acv.aruba_queue_case_owner_L7 IS NULL AND com.l7 IS NOT NULL)) OR
    (acv.aruba_queue_hpe_tower_manager != com.hpe_tower_manager OR (acv.aruba_queue_hpe_tower_manager IS NULL AND com.hpe_tower_manager IS NOT NULL)) OR
    (acv.aruba_queue_premium_team != com.premium_team OR (acv.aruba_queue_premium_team IS NULL AND com.premium_team IS NOT NULL)) OR
    (acv.aruba_queue_case_owner_country != com.country OR (acv.aruba_queue_case_owner_country IS NULL AND com.country IS NOT NULL)) OR
    (acv.aruba_queue_case_owner_state_city != com.state_city OR (acv.aruba_queue_case_owner_state_city IS NULL AND com.state_city IS NOT NULL)) OR
    (acv.aruba_queue_case_owner_location != com.location OR (acv.aruba_queue_case_owner_location IS NULL AND com.location IS NOT NULL)) OR
    (acv.aruba_queue_updated_primary_product_group != com.updated_primary_product_group OR (acv.aruba_queue_updated_primary_product_group IS NULL AND com.updated_primary_product_group IS NOT NULL)) OR
    (acv.aruba_queue_case_owner_manager_name != com.updated_case_owner_manager OR (acv.aruba_queue_case_owner_manager_name IS NULL AND com.updated_case_owner_manager IS NOT NULL)) 
  )
""")

# COMMAND ----------

case_owner_aruba_case_df.count()
display(case_owner_aruba_case_df.limit(5))

# COMMAND ----------

# DBTITLE 1,check if duplicates
from pyspark.sql.functions import col
duplicate_count = case_owner_aruba_case_df.groupBy("case_id").count().filter(col("count") > 1).count()

print(f"Duplicate case_id count: {duplicate_count}")

case_owner_aruba_case_df = case_owner_aruba_case_df.dropDuplicates(["case_id"])
case_owner_aruba_case_df.createOrReplaceTempView("case_owner_aruba_merge_view")

# COMMAND ----------

# DBTITLE 1,Cell 10
if case_owner_aruba_case_df.count() > 0:
  spark.sql("""
  MERGE INTO nnbu.nnbu_cases.jnpr_aruba_cases t
  USING case_owner_aruba_merge_view s
  ON t.case_id = s.case_id 
  WHEN MATCHED 
  THEN
    UPDATE SET
      t.case_owner_supervisor_email = s.case_owner_supervisor_email,
      t.aruba_queue_primary_product_category = s.aruba_queue_primary_product_category,
      t.aruba_case_owner_updated_queue = s.aruba_case_owner_updated_queue,
      t.aruba_case_owner_updated_queue_group = s.aruba_case_owner_updated_queue_group,
      t.aruba_queue_supervisor_name = s.aruba_queue_supervisor_name,
      t.aruba_queue_case_owner_role = s.aruba_queue_case_owner_role,
      t.aruba_queue_case_owner_sub_queue = s.aruba_queue_case_owner_sub_queue,
      t.aruba_queue_primary_product_group = s.aruba_queue_primary_product_group,
      t.aruba_queue_case_owner_org = s.aruba_queue_case_owner_org,
      t.aruba_queue_case_owner_L3 = s.aruba_queue_case_owner_L3,
      t.aruba_queue_case_owner_L4 = s.aruba_queue_case_owner_L4,
      t.aruba_queue_case_owner_L5 = s.aruba_queue_case_owner_L5,
      t.aruba_queue_case_owner_L6 = s.aruba_queue_case_owner_L6,
      t.aruba_queue_case_owner_L7 = s.aruba_queue_case_owner_L7,
      t.aruba_queue_hpe_tower_manager = s.aruba_queue_hpe_tower_manager,
      t.aruba_queue_premium_team = s.aruba_queue_premium_team,
      t.aruba_queue_case_owner_country = s.aruba_queue_case_owner_country,
      t.aruba_queue_case_owner_state_city = s.aruba_queue_case_owner_state_city,
      t.aruba_queue_case_owner_location = s.aruba_queue_case_owner_location,
      t.aruba_queue_updated_primary_product_group = s.aruba_queue_updated_primary_product_group,
      t.aruba_queue_case_owner_manager_name = s.aruba_queue_case_owner_manager_name,
      t.updt_dtm = current_timestamp()
  """)

# COMMAND ----------

pc_aruba_case_df = spark.sql("""
    SELECT
        acv.case_id,
        acv.aruba_product_category_derived_P1,
        acv.aruba_product_category_derived_P2 ,
        acv.aruba_product_category_derived_P3,
        acv.aruba_product_category_derived,
        p1.Product_Category AS aruba_product_category_derived_P1,
        p2.Product_Category AS aruba_product_category_derived_P2,
        p3.Product_Category AS aruba_product_category_derived_P3,
        COALESCE(p1.Product_Category, p2.Product_Category, p3.Product_Category) AS aruba_product_category_derived
    FROM nnbu.nnbu_cases.jnpr_aruba_cases acv 
   INNER JOIN (
        SELECT DISTINCT product_number_name, Product_Category
        FROM nnbu.sch_nnbu_rptng.vw_aruba_pc_mapping
        WHERE Priority = 'P1'
    ) p1
        ON p1.product_number_name = COALESCE(acv.aruba_case_product_number, acv.aruba_product_number)
    INNER JOIN (
        SELECT DISTINCT SFDC_Product_Group, Product_Category
        FROM nnbu.sch_nnbu_rptng.vw_aruba_pc_mapping
        WHERE Priority = 'P2'
    ) p2
        ON p2.SFDC_Product_Group = acv.aruba_product_group_engg
    INNER JOIN (
        SELECT DISTINCT product_number_name, Product_Category
        FROM nnbu.sch_nnbu_rptng.vw_aruba_pc_mapping
        WHERE Priority = 'P3'
    ) p3
        ON p3.product_number_name = COALESCE(acv.aruba_case_product_number, acv.aruba_product_number)
    WHERE acv.source = 'HPE'
     AND (
        (acv.aruba_product_category_derived_P1 != p1.Product_Category OR (acv.aruba_product_category_derived_P1 IS NULL AND p1.Product_Category IS NOT NULL))
        OR (acv.aruba_product_category_derived_P2 != p2.Product_Category OR (acv.aruba_product_category_derived_P2 IS NULL AND p2.Product_Category IS NOT NULL))
        OR (acv.aruba_product_category_derived_P3 != p3.Product_Category OR (acv.aruba_product_category_derived_P3 IS NULL AND p3.Product_Category IS NOT NULL))
        OR (acv.aruba_product_category_derived != COALESCE(p1.Product_Category, p2.Product_Category, p3.Product_Category) OR (acv.aruba_product_category_derived IS NULL AND COALESCE(p1.Product_Category, p2.Product_Category, p3.Product_Category) IS NOT NULL))
    ) 
""")



# COMMAND ----------

pc_aruba_case_df.count()
display(pc_aruba_case_df.limit(5))

# COMMAND ----------

from pyspark.sql.functions import col

duplicate_count = pc_aruba_case_df.groupBy("case_id").count().filter(col("count") > 1).count()
print(f"Duplicate case_id count: {duplicate_count}")

pc_aruba_case_df = pc_aruba_case_df.dropDuplicates(["case_id"])
pc_aruba_case_df.createOrReplaceTempView("case_pc_aruba_merge_view")

# COMMAND ----------

if pc_aruba_case_df.count() > 0:
    spark.sql("""
        MERGE INTO nnbu.nnbu_cases.jnpr_aruba_cases t
        USING case_pc_aruba_merge_view s
        ON t.case_id = s.case_id
        WHEN MATCHED
        THEN
          UPDATE SET
            t.aruba_product_category_derived_P1 = s.aruba_product_category_derived_P1,
            t.aruba_product_category_derived_P2 = s.aruba_product_category_derived_P2,
            t.aruba_product_category_derived_P3 = s.aruba_product_category_derived_P3,
            t.aruba_product_category_derived = s.aruba_product_category_derived,
            t.updt_dtm = current_timestamp()
    """)