#!/usr/bin/env python
import sys, os, getopt, re

from PatGlobalVars import *

import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

colhead_mol_smi = 'PROTAC_smi'
colhead_mol_name = 'PROTAC_id'
colhead_e3l_smi = 'E3_smi_stub'
colhead_linker_smi = 'Linker_smi_stub'
colhead_poi_smi = 'POI_smi_stub'
colhead_linker_bondspan = 'Linker_bondspan'
colhead_e3l_bondspan = 'E3_bondspan'
colhead_poi_bondspan = 'POI_bondspan'

def main(argv):
   global tsvrequest,verbose,numprod
   global libe3lfiles,libpoifiles,liblinkerfiles,tarmoltsv
   global linkerlibraw,e3llibraw,poilibraw
   global linkerlib,e3llib,poilib
   global colhead_mol_smi,colhead_mol_name,colhead_e3l_smi,colhead_poi_smi,colhead_linker_smi
   global dhac_up,dhac_down,dalogp_up,dalogp_down,dHBA_up,dHBA_down,dHBD_up,dHBD_down,dRB_up,dRB_down
   global dnArom_up,dnArom_down,dnRings_up,dnRings_down,dfCsp3_up,dfCsp3_down,dtPSA_up,dtPSA_down,dQED_up,dQED_down
   global prop_up_e3l,prop_down_e3l,prop_up_linker,prop_down_linker,prop_up_poi,prop_down_poi,prop_up_product,prop_down_product
   global e3l_props2use,linker_props2use,poi_props2use,product_props2use
   global mol_lb,mol_ub,mol_e3l_lb,mol_e3l_ub,mol_linker_lb,mol_linker_ub,mol_poi_lb,mol_poi_ub
   global e3l_prop_lb,e3l_prop_ub,linker_prop_lb,linker_prop_ub,poi_prop_lb,poi_prop_ub
   global maxprod

   maxprod = 1000000

   # Properties to be used in component/product filtering for this particular calculation
   e3l_props2use = ['HAC']
   linker_props2use = ['bondspan']
   poi_props2use = ['HAC','aLogP']
   product_props2use = ['HAC','aLogP','RotBonds']

   # default delta_property values for ALL AVAILABLE TPD COMPONENT PROPERTIES (applied upon reading/storing component libraries)
   prop_up_e3l = { 'bondspan':2, 'HAC':10, 'aLogP':1.0, 'RotBonds':2 }
   prop_down_e3l = { 'bondspan':2, 'HAC':6, 'aLogP':1.0, 'RotBonds':2 }
   prop_up_linker = { 'bondspan':3, 'HAC':8, 'aLogP':1.0, 'RotBonds':2 }
   prop_down_linker = { 'bondspan':3, 'HAC':8, 'aLogP':1.0, 'RotBonds':2 }
   prop_up_poi = { 'bondspan':2, 'HAC':6, 'aLogP':1.0, 'RotBonds':2 }
   prop_down_poi = { 'bondspan':2, 'HAC':6, 'aLogP':1.0, 'RotBonds':2 }
      # other properties:  HBA, HBD, nArom, nRings, fCsp3, tPSA

   # default delta_property values for ALL AVAILABLE TPD PRODUCT PROPERTIES (applied prior to write-out)
   prop_up_product = { 'HAC':6, 'aLogP':1.0, 'RotBonds':2 }
   prop_down_product = { 'HAC':6, 'aLogP':1.0, 'RotBonds':2 }
      # other properties:  HBA, HBD, nArom, nRings, fCsp3, QED, tPSA (hetatmratio?, stereo? pos/neg/net charges?) 

   libe3lfiles = []
   libpoifiles = []
   liblinkerfiles = []
   tsvrequest = ""
   tarmoltsv = ""
   verbose = False

   proc_commandline(argv)

   eprint("tsvrequest = {}".format(tsvrequest))
   eprint("verbose = {}".format(verbose))

   rdRequestTsv(tsvrequest)

   eprint("tarmoltsv = {}".format(tarmoltsv))
   eprint("liblinkerfiles = {}".format(liblinkerfiles))
   eprint("libe3lfiles = {}".format(libe3lfiles))
   eprint("libpoifiles = {}".format(libpoifiles))
   eprint("Component properties for filtration:")
   eprint("     E3L: {}".format(e3l_props2use))
   eprint("  Linker: {}".format(linker_props2use))
   eprint("     POI: {}".format(poi_props2use))
   eprint("Product properties for filtration: {}".format(product_props2use))

   eprint("Property drift bounds declared for degrader components, relative to target mol(s):")
   eprint("     E3L:")
   for prop in e3l_props2use:
      if( (prop in prop_up_e3l) and (prop in prop_down_e3l) ):
         eprint("          delta {}:  {} <= | => {}".format(prop,prop_up_e3l[prop],prop_down_e3l[prop]))
   eprint("  Linker:")
   for prop in linker_props2use:
      if( (prop in prop_up_linker) and (prop in prop_down_linker) ):
         eprint("          delta {}:  {} <= | => {}".format(prop,prop_up_linker[prop],prop_down_linker[prop]))
   eprint("     POI:")
   for prop in poi_props2use:
      if( (prop in prop_up_poi) and (prop in prop_down_poi) ):
         eprint("          delta {}:  {} <= | => {}".format(prop,prop_up_poi[prop],prop_down_poi[prop]))
   eprint("Property drift bounds declared for enumerated products, relative to target mol(s):")
   for prop in product_props2use:
      if( (prop in prop_up_product) and (prop in prop_down_product) ):
         eprint("     delta {}:  {} <= | => {}".format(prop,prop_up_product[prop],prop_down_product[prop]))

   rdTargetMols()

   e3l_prop_lb = {}
   e3l_prop_ub = {}
   linker_prop_lb = {}
   linker_prop_ub = {}
   poi_prop_lb = {}
   poi_prop_ub = {}
   eprint("Property bounds on degrader components:")
   eprint("     E3L:")
   for prop in e3l_props2use:
      if( (prop in prop_up_e3l) and (prop in prop_down_e3l) ):
         # E3L
         e3l_prop_lb[prop] = mol_e3l_lb[prop] - prop_down_e3l[prop]
         e3l_prop_ub[prop] = mol_e3l_ub[prop] + prop_up_e3l[prop]
         eprint("          {}: {} - {}".format(prop,e3l_prop_lb[prop],e3l_prop_ub[prop]))
   eprint("  Linker:")
   for prop in linker_props2use:
      if( (prop in prop_up_linker) and (prop in prop_down_linker) ):
         # Linker
         linker_prop_lb[prop] = mol_linker_lb[prop] - prop_down_linker[prop]
         linker_prop_ub[prop] = mol_linker_ub[prop] + prop_up_linker[prop]
         eprint("          {}: {} - {}".format(prop,linker_prop_lb[prop],linker_prop_ub[prop]))
   eprint("     POI:")
   for prop in poi_props2use:
      if( (prop in prop_up_poi) and (prop in prop_down_poi) ):
         # POI
         poi_prop_lb[prop] = mol_poi_lb[prop] - prop_down_poi[prop]
         poi_prop_ub[prop] = mol_poi_ub[prop] + prop_up_poi[prop]
         eprint("          {}: {} - {}".format(prop,poi_prop_lb[prop],poi_prop_ub[prop]))

   loadLibs()

   nprod = len(e3llibraw)*len(linkerlibraw)*len(poilibraw)
   eprint("Full potential (raw, un-filtered) enumeration: {} E3 binders X {} Linkers X {} POI binders = {} products".format(len(e3llibraw),len(linkerlibraw),len(poilibraw),nprod))

   prepLibs()

   #eprint("linkerlib:")
   #eprint("{}".format(linkerlib))

   nprod = len(e3llib)*len(linkerlib)*len(poilib)
   eprint("Full pruned enumeration: {} E3 binders X {} Linkers X {} POI binders = {} products".format(len(e3llib),len(linkerlib),len(poilib),nprod))


   enumFull()


   numprod = 0
   eprint("Finished. {} products written".format(numprod))

# ===========================================================================================

def enumFull():
   global e3llib,poilib,linkerlib,maxprod
   global goodenum,badenum

   print("SMILES,CpdName")

   goodenum = 0
   badenum = 0
   for e3smi in e3llib:
      for linkersmi in linkerlib:
         for poismi in poilib:
            growsmi = "{}.{}.{}".format(e3smi,linkersmi,poismi)
            growname = "{}_{}_{}".format(e3llib[e3smi]['molname'],linkerlib[linkersmi]['molname'],poilib[poismi]['molname'])
            #eprint("{}\t{}".format(growsmi,growname))
            try:
               m = Chem.MolFromSmiles(growsmi)
               prodcansmi = Chem.MolToSmiles(m,canonical=True,isomericSmiles=True,kekuleSmiles=False)

               # ---------------------------------------------------
               # Annotate and filter the product at this point
               # ---------------------------------------------------


   
               print("{}\t{}".format(prodcansmi,growname))
               goodenum += 1
            except:
               badenum += 1
            if(goodenum>=maxprod):
               eprint("Maximum number of products ({}) reached; exiting".format(maxprod))
               exit(0)

   eprint("Product enumeration: {} successes, {} failures".format(goodenum,badenum))

   return

# ===========================================================================================

def rdTargetMols():
   # Read-in the molecule(s) to target for this design request
   global libe3lfiles,libpoifiles,liblinkerfiles,tarmoltsv
   global e3llib,poilib,linkerlib
   global e3llibraw,poilibraw,linkerlibraw
   global colhead_mol_smi,colhead_mol_name,colhead_e3l_smi,colhead_poi_smi,colhead_linker_smi
   global mol_lb,mol_ub,mol_e3l_lb,mol_e3l_ub,mol_linker_lb,mol_linker_ub,mol_poi_lb,mol_poi_ub

   eprint("Reading Target Molecule(s) for this design request")

   eprint("tarmoltsv = {}".format(tarmoltsv))

   dfinmols = pd.read_csv(tarmoltsv,sep='\t')

   eprint("dfinmols.shape = {}".format(dfinmols.shape))
   eprint("dfinmols.head() = {}".format(dfinmols.head()))

   mol_e3l_ub = {}
   mol_e3l_lb = {}
   mol_linker_ub = {}
   mol_linker_lb = {}
   mol_poi_ub = {}
   mol_poi_lb = {}
   mol_ub = {}
   mol_lb = {}

   dfcollist = list(dfinmols.columns.values)
   for idx,row in dfinmols.iterrows():
      #eprint("   row = {}".format(row))
      for prop in product_props2use:
         colhead = "PROTAC_{}".format(prop)
         if(colhead in dfcollist):
            propval = row[colhead]
            if(prop in mol_ub):
               if(propval>mol_ub[prop]):
                  mol_ub[prop] = propval
            else:
               mol_ub[prop] = propval
            if(prop in mol_lb):
               if(propval<mol_lb[prop]):
                  mol_lb[prop] = propval
            else:
               mol_lb[prop] = propval
      for prop in e3l_props2use:
         # E3L component
         colhead = "E3_{}".format(prop)
         if(colhead in dfcollist):
            propval = row[colhead]
            if(prop in mol_e3l_ub):
               if(propval>mol_e3l_ub[prop]):
                  mol_e3l_ub[prop] = propval
            else:
               mol_e3l_ub[prop] = propval
            if(prop in mol_e3l_lb):
               if(propval<mol_e3l_lb[prop]):
                  mol_e3l_lb[prop] = propval
            else:
               mol_e3l_lb[prop] = propval
      for prop in linker_props2use:
         # Linker component
         colhead = "Linker_{}".format(prop)
         if(colhead in dfcollist):
            propval = row[colhead]
            if(prop in mol_linker_ub):
               if(propval>mol_linker_ub[prop]):
                  mol_linker_ub[prop] = propval
            else:
               mol_linker_ub[prop] = propval
            if(prop in mol_linker_lb):
               if(propval<mol_linker_lb[prop]):
                  mol_linker_lb[prop] = propval
            else:
               mol_linker_lb[prop] = propval
      for prop in poi_props2use:
         # POI component
         colhead = "POI_{}".format(prop)
         if(colhead in dfcollist):
            propval = row[colhead]
            if(prop in mol_poi_ub):
               if(propval>mol_poi_ub[prop]):
                  mol_poi_ub[prop] = propval
            else:
               mol_poi_ub[prop] = propval
            if(prop in mol_poi_lb):
               if(propval<mol_poi_lb[prop]):
                  mol_poi_lb[prop] = propval
            else:
               mol_poi_lb[prop] = propval


   eprint("Observed range of whole molecule properties amongst Target molecule(s):")
   for prop in mol_lb:
      eprint("        Mol: {}: {} - {}".format(prop,mol_lb[prop],mol_ub[prop]))
   eprint("Observed range of component molecule properties amongst Target molecule(s):")
   for prop in mol_e3l_lb:
      eprint("        E3L: {}: {} - {}".format(prop,mol_e3l_lb[prop],mol_e3l_ub[prop]))
   for prop in mol_linker_lb:
      eprint("     Linker: {}: {} - {}".format(prop,mol_linker_lb[prop],mol_linker_ub[prop]))
   for prop in mol_poi_lb:
      eprint("        POI: {}: {} - {}".format(prop,mol_poi_lb[prop],mol_poi_ub[prop]))

   return

# ===========================================================================================

def prepLibs():
   # Filter the E3L/Linker/POI libraries based upon component properties
   #    Also prepare the E3L/Linker/POI library SMILES for string concatenation-based enumeration
   global libe3lfiles,libpoifiles,liblinkerfiles,tarmoltsv
   global e3llib,poilib,linkerlib
   global e3llibraw,poilibraw,linkerlibraw

   smirks1 = "[Se]-[A,a:1]>>[2A,a:1]"
      # use atomic number 2 for the E3L-Linker bonds (indicated by Selenium faux atoms)
   smirks2 = "[Te]-[A,a:1]>>[3A,a:1]"
      # use atomic number 3 for the Linker-POI bonds (indicated by Tellurium faux atoms)

   rxn1 = AllChem.ReactionFromSmarts(smirks1)
   rxn2 = AllChem.ReactionFromSmarts(smirks2)

   # ============================
   #  E3L - Enum position #1
   # ============================
   i = 0
   enumpos = 1
   e3l_succs = 0
   e3l_fails = 0
   e3llib = {}
   for e3lsmi in e3llibraw:

      #if(e3l_succs<10):
      #   eprint("input e3lsmi = {}".format(e3lsmi))
      m = Chem.MolFromSmiles(e3lsmi)
      try:
         rxnout1 = rxn1.RunReactants((m, ))[0][0]
         Chem.SanitizeMol(rxnout1)
         presmi = Chem.MolToSmiles(rxnout1)
         #eprint("presmi = {}".format(presmi))
         tmp1 = re.sub(r'[0-9]+\]',']',presmi)
         prepsmi = re.sub(r'H\]',']',tmp1)
         #eprint("prepsmi = {}".format(prepsmi))
         fixsmi = fixSmiles(prepsmi,enumpos)
         if(fixsmi not in e3llib):
            e3llib[fixsmi] = e3llibraw[e3lsmi]
         if(e3l_succs<10):
            eprint("E3L fixsmi = {}".format(fixsmi))
         e3l_succs += 1
      except:
         e3l_fails += 1
      i += 1
      eprint("")
   eprint("E3L binder smiles preparation:  {} successes, {} failures".format(e3l_succs,e3l_fails))


   # ============================
   #  Linkers - Enum position #2
   # ============================
   i = 0
   enumpos = 2
   linker_succs = 0
   linker_fails = 0
   linkerlib = {}
   for linkersmi in linkerlibraw:
      if(linker_succs<10):
         eprint("input linkersmi = {}".format(linkersmi))
      m = Chem.MolFromSmiles(linkersmi)
      rxn1 = AllChem.ReactionFromSmarts(smirks1)
      rxn2 = AllChem.ReactionFromSmarts(smirks2)
      try:
         rxnout1 = rxn1.RunReactants((m, ))[0][0]
         Chem.SanitizeMol(rxnout1)
         rxnout2 = rxn2.RunReactants((rxnout1, ))[0][0]
         Chem.SanitizeMol(rxnout2)
         presmi = Chem.MolToSmiles(rxnout2)
         tmp1 = re.sub(r'[0-9]+\]',']',presmi)
         prepsmi = re.sub(r'H\]',']',tmp1)
         fixsmi = fixSmiles(prepsmi,enumpos)
         if(fixsmi not in linkerlib):
            # add this linker from linkerlibraw to linkerlib
            linkerlib[fixsmi] = linkerlibraw[linkersmi]

         if(linker_succs<10):
            #eprint("Linker prepsmi = {}".format(prepsmi))
            eprint("Linker fixsmi = {}".format(fixsmi))
         linker_succs += 1
      except:
         linker_fails += 1
      i += 1
   eprint("Linker smiles preparation:  {} successes, {} failures".format(linker_succs,linker_fails))

   # ============================
   #  POIs - Enum position #3 
   # ============================
   i = 0
   enumpos = 3
   poi_succs = 0
   poi_fails = 0
   poilib = {}
   for poismi in poilibraw:
      if(poi_succs<10):
         eprint("input poismi = {}".format(poismi))
      m = Chem.MolFromSmiles(poismi)
      try:
         rxnout2 = rxn2.RunReactants((m, ))[0][0]
         Chem.SanitizeMol(rxnout2)
         presmi = Chem.MolToSmiles(rxnout2)
         tmp1 = re.sub(r'[0-9]+\]',']',presmi)
         prepsmi = re.sub(r'H\]',']',tmp1)
         fixsmi = fixSmiles(prepsmi,enumpos)
         if(fixsmi not in poilib):
            poilib[fixsmi] = poilibraw[poismi]

         if(poi_succs<10):
            #eprint("POI prepsmi = {}".format(prepsmi))
            eprint("POI fixsmi = {}".format(fixsmi))
         poi_succs += 1
      except:
         poi_fails += 1
      i += 1
   eprint("POI binder smiles preparation:  {} successes, {} failures".format(poi_succs,poi_fails))


   return

# ===========================================================================================

def fixSmiles(insmi,ipos):

   #eprint("insmi = {} and ipos = {}".format(insmi,ipos))

   osmi = insmi

   # ================================
   # Fix intra-BB ring labels first
   # ================================

   if(ipos>0):

      #eprint("Fixing intra-BB ring labels first ...")

      # Two-letter elements of relevance:  Cl, Br, Si, Na, Mg, Ca, Al, and faux placeholders Se, Si, Ge

      #smi = "[3c]12cc(CC)cnc2[1C]N(COC)[2C]1"
      #name = "debug"
      #ipos = 2

      tok = re.findall(r'\[[1-9]*[a-zA-Z][a-z]?[@]*[H]?[-+]?\][1-9]+|[A-Z][l,r,i,a,g]?[1-9]+|[a-z][1-9]+',insmi)
      split = re.split(r'\[[1-9]*[a-zA-Z][a-z]?[@]*[H]?[-+]?\][1-9]+|[A-Z][l,r,i,a,g]?[1-9]+|[a-z][1-9]+',insmi)

      #if(name=="debug" or name=="1124" or name=="1" or name=="816"):
      #   eprint("BEFORE, name = {} insmi = {}".format(name,insmi))
      #   eprint("   Ring labels findall:")
      #   eprint("   {}".format(tok))
      #   eprint("   split:")
      #   eprint("   {}".format(split))

      osmi = ""
      for i in range(0,len(split)):
         osmi += "{}".format(split[i])
         if(i<(len(split)-1)):
            # adjust ring index based upon ipos: e.g. if ipos = 2 and the ring index is 4, then the new ring index is %24
            iring = re.findall(r'[1-9]+$',tok[i])
            isplit = re.split(r'[1-9]+$',tok[i])
            jring = re.findall(r'[1-9]',iring[0])
            ringstr = "{}".format(isplit[0])
            for jay in range(0,len(jring)):
               kring = 10*ipos + int(jring[jay])
               if(ringstr==""):
                  ringstr = "%{}".format(kring)
               else:
                  ringstr += "%{}".format(kring)
            #if(name=="debug" or name=="1124" or name=="1" or name=="816"):
            #   eprint("   iring = {} and isplit = {}".format(iring,isplit))
            #   eprint("   i = {} and iring = {} and tok[i] = {}".format(i,iring,tok[i]))
            #   eprint("   ringstr = {}".format(ringstr))
            osmi += "{}".format(ringstr)

      #if(name=="debug" or name=="1124" or name=="1" or name=="816"):
      #   eprint("AFTER fixing intra-BB ring labels: osmi = |{}|".format(osmi))

   # ==============================
   # Fix isotopic bond labels next
   # ==============================

   #eprint("")
   #eprint("findall isotopic labels in {}:".format(osmi))
   tok = re.findall(r'\[[1-9]?[1-9][a-zA-Z@]+\]',osmi)

   #eprint("splitting osmi={}:".format(osmi))
   split = re.split(r'\[[1-9]?[1-9][a-zA-Z@]+\]',osmi)

   osmi = ""
   for i in range(0,len(split)):
      osmi += "{}".format(split[i])
      if(i<(len(split)-1)):
         # adjust ring index based upon ipos: e.g. if ipos = 2 and the ring index is 4, then the new ring index is %24
         #eprint("     dynamic osmi = {} ... now to add the ring index nomenclature for tok[i] = {}".format(osmi,tok[i]))
         iring = re.findall(r'[1-9]',tok[i])
         ringstr = ""
         for jay in range(0,len(iring)):
            jring = 10*int(iring[jay])
            if(ringstr==""):
               ringstr = "%{}".format(jring)
            else:
               ringstr += "%{}".format(jring)
         nmatches = len(re.findall(r'[H@]',tok[i]))
         ringsplit = re.split(r'[1-9]?[1-9]',tok[i])
         if(nmatches==0):
            elem = ringsplit[1].replace(']','')
            otok = "{}{}".format(elem,ringstr)
            osmi += "{}".format(otok)
            #eprint("   ... and ringstr = {}, elem = {}, otok = {} ==> osmi = {}".format(ringstr,elem,otok,osmi))
         elif(nmatches>0):
            elemstr = ""
            for k in range(0,len(ringsplit)):
               elemstr += ringsplit[k]
            otok = "{}{}".format(elemstr,ringstr)
            osmi += "{}".format(otok)
            #eprint("   ... and ringstr = {}, elemstr = {}, otok = {} ==> osmi = {}".format(ringstr,elemstr,otok,osmi))

   #eprint("after, osmi = {}".format(osmi))

   return osmi

# ===========================================================================================

def loadLibs():
   # Load each set of component structures from file one by one
   #    Upon loading a component, apply COMPONENT level property filters & save in memory only those that pass
   global libe3lfiles,libpoifiles,liblinkerfiles,tarmoltsv
   global e3llib,poilib,linkerlib
   global e3llibraw,poilibraw,linkerlibraw
   global colhead_mol_smi,colhead_mol_name,colhead_e3l_smi,colhead_poi_smi,colhead_linker_smi
   global prop_up_e3l,prop_down_e3l,prop_up_linker,prop_down_linker,prop_up_poi,prop_down_poi,prop_up_product,prop_down_product
   global e3l_props2use,linker_props2use,poi_props2use,product_props2use
   global mol_lb,mol_ub,mol_e3l_lb,mol_e3l_ub,mol_linker_lb,mol_linker_ub,mol_poi_lb,mol_poi_ub
   global e3l_prop_lb,e3l_prop_ub,linker_prop_lb,linker_prop_ub,poi_prop_lb,poi_prop_ub


   # -------------
   #  E3L binder 
   # -------------
   ie3l = 1
   e3llibraw = {}
   lib_e3l_keepers = 0
   lib_e3l_rejects = 0
   for e3ltsv in libe3lfiles:
      eprint("reading e3l from {} ...".format(e3ltsv))
      dfe3l = pd.read_csv(e3ltsv,sep='\t')
      #eprint("len(dfe3l) = {}".format(len(dfe3l)))
      #eprint("dfe3l.head() = {}".format(dfe3l.head()))
      dfcollist = list(dfe3l.columns.values)
      #eprint("dfe3l.columns = {}".format(dfe3l.columns))
      if(colhead_e3l_smi in dfcollist):
         for idx,row in dfe3l.iterrows():
            smi = row[colhead_e3l_smi]
            if(smi not in e3llibraw):
               if(colhead_mol_name in dfcollist):
                  molname = row[colhead_mol_name]
               else:
                  molname = "E3l{}".format(ie3l)
                  ie3l += 1
               libdict = {}
               outofbounds = False
               libdict['molname'] = molname
               for propname in e3l_props2use:
                  prop_colhead = "E3_{}".format(propname)
                  if(prop_colhead in dfcollist):
                     propval = row[prop_colhead]
                  else:
                     propval = -99
                  if( (propval < e3l_prop_lb[propname]) or (propval > e3l_prop_ub[propname]) ):
                     outofbounds = True
                  libdict[propname] = propval
               if(not outofbounds):
                  # properties would also belong in liblist
                  e3llibraw[smi] = libdict
                  lib_e3l_keepers += 1
               else:
                  lib_e3l_rejects += 1

   j = 0
   for smi in e3llibraw:
      if(j<20):
         eprint("   smi = {}".format(smi))
         eprint("      e3llibraw = {}".format(e3llibraw[smi]))
      j += 1

   eprint("E3L binders library loading: {} E3L binders to be used; {} rejected based upon properties".format(lib_e3l_keepers,lib_e3l_rejects))

   # -------------
   #  LINKERS 
   # -------------
   ilinker = 1
   linkerlibraw = {}
   lib_linker_keepers = 0
   lib_linker_rejects = 0
   for linktsv in liblinkerfiles:
      eprint("reading linkers from {} ...".format(linktsv))
      dflinker = pd.read_csv(linktsv,sep='\t')
      #eprint("len(dflinker) = {}".format(len(dflinker)))
      #eprint("dflinker.head() = {}".format(dflinker.head()))
      dfcollist = list(dflinker.columns.values)
      #eprint("dflinker.columns = {}".format(dflinker.columns))
      if(colhead_linker_smi in dfcollist):
         for idx,row in dflinker.iterrows():
            smi = row[colhead_linker_smi]
            if(smi not in linkerlibraw):
               if(colhead_mol_name in dfcollist):
                  molname = row[colhead_mol_name]
               else:
                  molname = "Linker{}".format(ilinker)
                  ilinker += 1
               libdict = {}
               outofbounds = False
               libdict['molname'] = molname
               for propname in linker_props2use:
                  prop_colhead = "Linker_{}".format(propname)
                  if(prop_colhead in dfcollist):
                     propval = row[prop_colhead]
                  else:
                     propval = -99
                  if( (propval < linker_prop_lb[propname]) or (propval > linker_prop_ub[propname]) ):
                     outofbounds = True
                  libdict[propname] = propval
               if(not outofbounds):
                  # properties would also belong in liblist
                  linkerlibraw[smi] = libdict
                  lib_linker_keepers += 1
               else:
                  lib_linker_rejects += 1

   j = 0
   for smi in linkerlibraw:
      if(j<20):
         eprint("   smi = {}".format(smi))
         eprint("      linkerlibraw = {}".format(linkerlibraw[smi]))
      j += 1

   eprint("Linker library loading: {} linkers to be used; {} linkers rejected based upon properties".format(lib_linker_keepers,lib_linker_rejects))

   # -------------
   #  POI 
   # -------------
   ipoi = 1
   poilibraw = {}
   lib_poi_keepers = 0
   lib_poi_rejects = 0
   for poitsv in libpoifiles:
      eprint("reading pois from {} ...".format(poitsv))
      dfpoi = pd.read_csv(poitsv,sep='\t')
      #eprint("len(dfpoi) = {}".format(len(dfpoi)))
      #eprint("dfpoi.head() = {}".format(dfpoi.head()))
      dfcollist = list(dfpoi.columns.values)
      #eprint("dfpoi.columns = {}".format(dfpoi.columns))
      if(colhead_poi_smi in dfcollist):
         for idx,row in dfpoi.iterrows():
            smi = row[colhead_poi_smi]
            if(smi not in poilibraw):
               if(colhead_mol_name in dfcollist):
                  molname = row[colhead_mol_name]
               else:
                  molname = "POI{}".format(ipoi)
                  ipoi += 1
               libdict = {}
               outofbounds = False
               libdict['molname'] = molname
               for propname in poi_props2use:
                  prop_colhead = "POI_{}".format(propname)
                  if(prop_colhead in dfcollist):
                     propval = row[prop_colhead]
                  else:
                     propval = -99
                  if( (propval < poi_prop_lb[propname]) or (propval > poi_prop_ub[propname]) ):
                     outofbounds = True
                  libdict[propname] = propval
               if(not outofbounds):
                  # properties would also belong in liblist
                  poilibraw[smi] = libdict
                  lib_poi_keepers += 1
               else:
                  lib_poi_rejects += 1

   j = 0
   for smi in poilibraw:
      if(j<20):
         eprint("   smi = {}".format(smi))
         eprint("      poilibraw = {}".format(poilibraw[smi]))
      j += 1

   eprint("POI library loading: {} POIs to be used; {} rejected based upon properties".format(lib_poi_keepers,lib_poi_rejects))


   eprint("{} E3l binders to be used in enumeration".format(len(e3llibraw)))
   eprint("{} Linkers to be used in enumeration".format(len(linkerlibraw)))
   eprint("{} POI binders to be used in enumeration".format(len(poilibraw)))

   return

# ===========================================================================================

def rdRequestTsv(reqtsv):
   global tsvrequest,verbose,numprod
   global libe3lfiles,libpoifiles,liblinkerfiles,tarmoltsv
   
   eprint("hey, reqtsv = {}".format(reqtsv))
   rfile = open(reqtsv,'r')
   for rline in rfile:
      rline = rline.strip()
      eprint("rline = {}".format(rline))
      rlist = rline.split("\t")
      rlistlen = len(rlist)
      eprint("rlistlen = {} and rlist[0] = {}".format(rlistlen,rlist[0]))
      if(rlistlen>1):
         rkey = str(rlist[0])
         rvalue = str(rlist[1])
         eprint("hey, rkey = |{}| and rvalue = |{}|".format(rkey,rvalue))
         if(rkey == "tarmols"):
            tarmoltsv = rvalue
         elif(rkey == "linkerlib"):
            if(rvalue not in liblinkerfiles):
               liblinkerfiles.append(rvalue)
         elif(rkey == "e3llib"):
            if(rvalue not in libe3lfiles):
               libe3lfiles.append(rvalue)
         elif(rkey == "poilib"):
            if(rvalue not in libpoifiles):
               libpoifiles.append(rvalue)


   if(len(liblinkerfiles)==0):
      liblinkerfiles.append(tarmoltsv)
      eprint("No Linker library file(s) provided- Linker(s) will be taken from input molecule(s)")
   else:
      eprint("Linkers being taken from file(s): {}".format(liblinkerfiles))
   if(len(libe3lfiles)==0):
      libe3lfiles.append(tarmoltsv)
      eprint("No E3L binder library file(s) provided- E3L binder(s) will be taken from input molecule(s)")
   else:
      eprint("E3L binders being taken from file(s): {}".format(libe3lfiles))
   if(len(libpoifiles)==0):
      libpoifiles.append(tarmoltsv)
      eprint("No POI binder library file(s) provided- POI binders will be taken from input molecule(s)")
   else:
      eprint("POI binders being taken from file(s): {}".format(libpoifiles))

   rfile.close()

   return

# ===========================================================================================

def proc_commandline(argv):
   global tsvrequest,verbose

   verbose = False
   tsvrequest = ""

   usage  = "=============================================================================================\n"
   usage += " caddbot.py\n"
   usage += " Produces designed in silico degrader structures to fulfill a single design request from a therapeutic program\n"
   usage += "\n";
   usage += " USAGE: bash2py.bash caddbot.py -i CaddbotRequestInput.tsv <OPTIONS> \n"
   usage += "\n";
   usage += " <OPTIONS>\n"
   usage += "      <-vERBOSE> ==> Extra verbose output to stderr\n"
   usage += "=============================================================================================\n"

   if(len(sys.argv)<2):
      print(usage)
      sys.exit(0)

   try:
      opts, args = getopt.getopt(argv,"i:v")
   except getopt.GetoptError:
      print(usage)
      sys.exit(2)

   for opt, arg in opts:
      if(opt == '-i'):
         tsvrequest = arg.strip()
      elif(opt == '-v'):
         verbose = True

   return

# ===========================================================================================
if(__name__ == "__main__"):
   main(sys.argv[1:])
