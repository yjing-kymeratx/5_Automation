#!/usr/bin/env python

import sys, os, getopt
import csv
import pandas as pd

from PatGlobalVars import *

d360columns = ["PROTAC_id","Decomp_Flag","Decomp_Level","E3_smi","E3_smi_stub","Linker_smi","Linker_smi_stub","POI_smi","POI_smi_stub","Lib_E3_target","Lib_POI_target","KymE3L_Chemotype","KymPOI_Chemotype","E3_flag","POI_flag"]

# Optional step: Remove warnings thrown by invalid SSL certificate.
import warnings
warnings.filterwarnings('ignore')

# server = "10.3.30.118"
# port = "5432"
# uname = "cadd_rw"
# psswd = "k5$ypghw$X!RRC^)"
# dbname = "compound_decomposition"

import psycopg2
from psycopg2 import Error

def main(argv):
   global outdir,outtsv,qid

   cwd = os.getcwd()
   updatefreq = 1000

   start_datetimestr = datetime.datetime.now()
   startdate = "{}/tpdnav_d360push.startdate".format(tpdnav_rootdir)
   cmd = "echo {} > {}; chmod 775 {}".format(start_datetimestr,startdate,startdate)
   eprint("cmd = {}".format(cmd))
   os.system(cmd)
   
   eprint("Reading data from {}".format(tpdnav_Kymera_D360))
   tpdf = pd.read_csv(tpdnav_Kymera_D360)
   print("tpdf.head():")
   print(tpdf.head())
   eprint("len(tpdf.index) = {}".format(len(tpdf.index)))

   try:
      eprint("Trying to connect ...")
      conn = psycopg2.connect(user="cadd_rw", password="k5$ypghw$X!RRC^)", host="10.3.30.118", port="5432", database="compound_decomposition")
      cursor = conn.cursor()

      # blow away all rows of data in the kymera_protac_decomposition table
      dquery = """delete from prod.kymera_protac_decomposition"""
      eprint("dquery = {}".format(dquery))
      cursor.execute(dquery)
      conn.commit()
      eprint("Just blew away all rows of data in kymera_protac_decomposition table ...\n")

      eprint("Bulk uploading the data in file {} to the D360 postgres DB ...".format(tpdnav_Kymera_D360))

      sqlcmd = "COPY prod.kymera_protac_decomposition from STDIN DELIMITER ',' CSV HEADER;"
      eprint("sqlcmd = {}".format(sqlcmd))
      #cursor.copy_expert(sqlcmd, open("/fsx/data/AUTOMATION/TPDNav/KymeraMols.tpdecomp.D360.csv", "r"))
      cursor.copy_expert(sqlcmd, open(tpdnav_Kymera_D360, "r"))
      conn.commit()
      conn.close()

   except (Exception, Error) as error:
      eprint("Error while connecting to PostgreSQL",error)

   end_datetimestr = datetime.datetime.now()
   enddate = "{}/tpdnav_d360push.enddate".format(tpdnav_rootdir)
   cmd = "echo {} > {}; chmod 775 {}".format(end_datetimestr,enddate,enddate)
   eprint("cmd = {}".format(cmd))
   os.system(cmd)

   eprint("Done.")

   sys.stdout.flush()
   sys.stderr.flush()

# ===========================================================================================
if(__name__ == "__main__"):
   main(sys.argv[1:])
