#!pip install biopython
#!pip install networkx
#!pip install seaborn
#!pip install griddataformats

import numpy as np
#immort random number from np
from numpy import random
from Bio.PDB import PDBParser
from gridData import Grid
rng= random.default_rng()
import matplotlib.pyplot as plt
import os
import numpy as np
import networkx as nx
import seaborn as sns
from sklearn.manifold import MDS
from scipy.ndimage import fourier_gaussian
from mpl_toolkits.axes_grid1 import make_axes_locatable

hbar_eV = 6.582119569e-16 #eV*s
kbT_eV = 8.617333262145E-5*300 #eV

sns.set_style("whitegrid")
sns.set_context("paper")

atomMass = {'C': 12.0107, 'H': 1.00794,
            'N': 14.0067, 'O': 15.9994, 'S': 32.065}

atomRadius = {'C': 0.77, 'H': 0.37,
              'N': 0.75, 'O': 0.73, 'S': 1.02}  # https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)

def LoadAminoCenterOfMass(pdb_file,redoxEnergies,startSites,exitSites,verbose=False ):
    """Loads a pbd file, a potential file and a charge file and returns the center of mass of each amino acid

    Keyword arguments:
    pdb_file -- the pdb file to load.  Should have the same names a the optional potential and charge files (_apbs_potential.dx and .charge)
    redoxEnergies -- a dictionary of redox energies for each redox cofactor amino acid
    startElectronInjection -- the amino acid connected to electrode (dictionary with keys: residue, model, chain, index)
    exitElectronInjection-- the amino acid connected to electrode (dictionary with keys: residue, model, chain, index)
    """  
    parser = PDBParser()
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure('x', pdb_file)
    

    hasPotential, g, origin,delta= loadDXFile(pdb_file, verbose)
     
    #get list of residues
    atom_centerOfMasss = []
    for model in structure:
        for chain in model:
            for residue in chain.get_residues():
                #get the atoms for this residue
                oxygens,potentials,charge=GetSideAtoms(residue,hasPotential,g,origin,delta)
                    
                if hasPotential:
                        coordsA =np.array( residue.center_of_mass())
                        index=((coordsA-origin)/delta).astype(int)
                        for i in range(len(index)):
                            if index[i]<0:
                                index[i]=0
                            if index[i]>=g.grid.shape[i]:
                                index[i]=g.grid.shape[i]-1
                        pot =( g.grid[index[0],index[1],index[2]]) #supposed to be in kt/e
                else:
                    pot =0
                
                
                atom_com = {'amino':residue.get_resname(),
                            'i':len(atom_centerOfMasss),
                            'model':model.get_id(),
                            'chain':chain.get_id(),
                            'index':residue.get_id()[1],
                            'centerOfMass':np.array(residue.center_of_mass())/10.0,
                            'redoxPoints':oxygens,
                            'potenial_EV': pot *0.0259/10,  
                            'avepotenial_APBS': np.mean(potentials) ,  
                            'charge':charge, 
                            'startInject':False,
                            'endEject':False,
                            'std':0,
                            'redoxEnergy_EV': -10000,
                            'neighbors':[]}
                
                if atom_com['amino'] in redoxEnergies:
                    atom_com['redoxEnergy_EV'] = redoxEnergies[atom_com['amino']]  
                atom_centerOfMasss.append(atom_com)
                    
            
     # select the amino acids that are are redox cofactors
    activeAminos = []
    for amino in atom_centerOfMasss:
        if amino['amino'] in redoxEnergies:
            activeAminos.append(amino)
            
    findRedoxCofactors(activeAminos,startSites,exitSites, verbose) 
    if verbose:      
        chains = list(set([x['chain'] for x in atom_centerOfMasss]))
        for chain in chains:
            print(f"{chain} has {len( [x['amino'] for x in atom_centerOfMasss if x['chain']==chain ] )} amino acids with {len( [x['amino'] for x in activeAminos if x['chain']==chain ] )} redox cofactors")                     
        
    
    loadChargeFile(pdb_file,atom_centerOfMasss, verbose )
    loadSTDFile(pdb_file,atom_centerOfMasss, verbose )
        
    return atom_centerOfMasss,activeAminos 

def GetSideAtoms(residue,hasPotential,g,origin,delta):
    charge = 0 
    oxygens=[]
    potentials = [] 
    atomCount =0
    for atom in  residue.get_atoms():
        coords =np.array( atom.get_coord())/10.0 #convert to nm
        chr=atom.get_charge()
        if chr:
            charge += chr
            
        #load the APBS potential and get the potential at this atom
        if hasPotential:
            coordsA =np.array( atom.get_coord())
            index=((coordsA-origin)/delta).astype(int)
            for i in range(len(index)):
                if index[i]<0:
                    index[i]=0
                if index[i]>=g.grid.shape[i]:
                    index[i]=g.grid.shape[i]-1
            potentials .append( g.grid[index[0],index[1],index[2]])
    
        element = atom.get_id()[0]
        #oxygen and thiols are used to determine jump distance
        if element == 'O' or element == 'S' or element == 'N' or (element == 'C' and atom.get_name() != 'CA'):
            oxygens.append([element,coords])
        atomCount +=1
    return oxygens,potentials,charge

def findRedoxCofactors(activeAminos,startSites,exitSites, verbose=False):
    for i in range(len(activeAminos)):
        for starts in startSites:  
            if  activeAminos[i]['amino'] == starts['residue'] and  \
                activeAminos[i]['model'] == starts['model'] and  \
                activeAminos[i]['chain'] == starts['chain'] and  \
                activeAminos[i]['index'] == starts['index'] :
                    if verbose:
                        print('Found Injection Node', i)
                    starts['aminoIndex'] = i
                    activeAminos[i]['startInject']=True
                
                
        for exits in exitSites:
            if  activeAminos[i]['amino'] == exits['residue'] and  \
                activeAminos[i]['model'] == exits['model'] and  \
                activeAminos[i]['chain'] == exits['chain'] and  \
                activeAminos[i]['index'] == exits['index'] :
                    if verbose:
                        print('Found Exit Node', i)
                    exits['aminoIndex'] =  i
                    activeAminos[i]['endEject']=True
                
     

def loadDXFile(pdb_file, verbose=False):
    g=None
    origin=None
    delta=None
    if os.path.exists(pdb_file.replace('.pdb', '.dx')):
        if verbose:
            print('dx file found')
        hasPotential = True
        g = Grid(pdb_file.replace('.pdb', '.dx'))
        origin,delta = g.origin, g.delta 
    elif os.path.exists(pdb_file.replace('.pdb', '_apbs_potential.dx')):
        if verbose:
            print('dx file found')
        hasPotential = True
        g = Grid(pdb_file.replace('.pdb', '_apbs_potential.dx'))
        origin,delta = g.origin, g.delta
    else:
        hasPotential = False
        if verbose:
            print('no dx file found')    
    return hasPotential, g, origin,delta

def loadSTDFile(pdb_file,atom_centerOfMasss, verbose=False):
    stdFile = pdb_file.replace('.pdb', '.std')  
    print('Find vibrations at ' + stdFile)
    if os.path.exists(stdFile):
        if verbose:
            print('Found std file:' , stdFile)
        with open(stdFile) as f:
            lines = f.readlines()
        
        for line in lines:
            atom, std = line.strip().split('\t')
            atom, std = int(atom.strip()    )-1, float(std.strip()  )
            atom_centerOfMasss[atom]['std'] = std     
    
def loadChargeFile(pdb_file,atom_centerOfMasss, verbose=False):
    chargeFile = pdb_file.replace('.pdb', '.charge')  
    print('Find ChargeFile at ' + chargeFile)
    if os.path.exists(chargeFile):
        if verbose:
            print('Found charge file:' , chargeFile)
        with open(chargeFile) as f:
            lines = f.readlines()
        
        for line in lines:
            atom, charge = line.split()
            atom,charge = int(atom.strip()    )-1, float(charge.strip()  )
            atom_centerOfMasss[atom]['charge'] = charge        
    

def getNeighborOccupancy(active1,active2, atoms_COM, vanDerWaals_guess_radius_nm=0.4):
    """Determine if the electron is tunneling through protein media or through other substances
    draw a line between atom and its neighbors, check ever .2 angstroms if there is a coord that is less than radius from the segment location

    Keyword arguments:
    active1 -- the donor COM amino acid (dictionary)
    active2 -- the accepter COM amino acid (dictionary)
    atoms_COM -- the list of all amino acids (list of dictionaries)
    vanDerWaals_guess_radius_nm  -- the radius to use for the van der waals radius of the amino acid 
    
    todo:use the redox points to determine the radius
    """  
     
    startAtom = active1['centerOfMass']
    endAtom = active2['centerOfMass']
    slope = endAtom - startAtom
    
    length = np.linalg.norm(slope)
    
    if length==0:
        return 1
    else:
        #get the points along the line
        stops = startAtom + slope * np.arange(0, length, .2)[:, None] / length
        
        #calculate the percent of stops that are within the van der waals radius of the neighbor
        percent =0
        for d in range(len(stops)):
            for atom in atoms_COM:
                coord = atom['centerOfMass']
                dd= np.linalg.norm(coord - stops[d])
                if dd< vanDerWaals_guess_radius_nm:
                    percent += 1
                    break
                
        percent = percent / len(stops)
         
        return percent   

def CalculatePotentials(atom_COM,activeAminos,startIjectNode,endIjectNode,appliedVoltage_V, verbose =False):
    """#determine the driving voltage for for moving the electron through the molecule with respect to the partial charges.
    Args:
        atom_COM (_type_): _description_
        activeAminos (_type_): _description_
        startIjectNode (_type_): _description_
        endIjectNode (_type_): _description_
        appliedVoltage_V (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    field = activeAminos[endIjectNode['aminoIndex']]['centerOfMass']-activeAminos[startIjectNode['aminoIndex']]['centerOfMass']
    L= np.linalg.norm(field)
    field = field/np.linalg.norm(field) *  0.02042  * appliedVoltage_V 
    fieldZero = activeAminos[startIjectNode['aminoIndex']]['centerOfMass']
    if verbose:
        print(f"Driving force for electron transfer in molecule is { np.linalg.norm(field)*L:.2f} V/nm")    

    for i in range(len(atom_COM)):
        voltage =0 
        for j in range(len(atom_COM)):
            otherCharge = atom_COM[j]['charge']              #in electrons
            if otherCharge!=0:
                distance = np.linalg.norm(atom_COM[i]['centerOfMass']-atom_COM[j]['centerOfMass']) #distance in nm
                if distance < .4: #assume that we are not tunneling to the atom with the charge.
                    distance = .4
                voltage += .0144/2 * otherCharge / distance  #voltage in V
                
        distance = (atom_COM[i]['centerOfMass']-fieldZero) #distance in nm
        atom_COM[i]['totalPotential'] = voltage + np.dot( field ,distance)
    return field, fieldZero

def sortPaths(shortestPaths, weight):
    transitTimes = [x[weight] for x in shortestPaths]
    fastToSlow = np.argsort(transitTimes).astype(int).tolist()
    shortestPaths =[ shortestPaths[x] for x in fastToSlow]
    return shortestPaths

def CalculatePaths(G,activeAminos, startIjectNode,endIjectNode, weight ):  
    """use dijkstra's algorithm to find the shortest path between the start and end nodes
    randomize the paths by forcing the path through all the nodes in the protein, but allowing the algorythm to get the optimal solutions

    Args:
        G (graph): graph with edges that have a weight of time
        activeAminos (_type_): _description_
        startIjectNode (_type_): _description_
        endIjectNode (_type_): _description_

    Returns:
        _type_: list of dictionary of paths sorted by the fastest first.
    """

    #get the shortest path in a few different ways.  Sometimes each method will return a variety of paths if they are close in length
    shortestPaths_static = list(nx.all_shortest_paths(G, source=startIjectNode, target=endIjectNode, weight=weight, method='dijkstra'))
    shortestPaths_static.extend ( list(nx.all_shortest_paths(G, source=startIjectNode, target=endIjectNode, weight=weight, method='bellman-ford')))

    for node in list(G.nodes) :
        if node != startIjectNode and node != endIjectNode:
            #get a few that are only based on the jumps
            try:
                jumpPaths = list(nx.all_shortest_paths(G, source=startIjectNode, target=node, weight=weight, method='dijkstra'))
                jumpPaths.extend ( list(nx.all_shortest_paths(G, source=startIjectNode, target=node, weight=weight, method='bellman-ford')))
                
                endPaths =  list(nx.all_shortest_paths(G, source=node, target=endIjectNode, weight=weight, method='dijkstra'))
                endPaths.extend ( list(nx.all_shortest_paths(G, source=node, target=endIjectNode, weight=weight, method='bellman-ford')))

                #combine the paths
                for jumpPath in jumpPaths:
                    for endPath in endPaths:
                        path = jumpPath + endPath[1:]
                        shortestPaths_static.append(path)
            except Exception as e:
                print(f"{e} : {node} : {G.nodes[node]['label']}")
                 
                pass

    shortestPaths = []
    for path in shortestPaths_static:
        newCoords = []
        for index in path:
            coords=(activeAminos[index]['centerOfMass'])
            newCoords.append(coords)
            
        weight_gaps =[]
        dists=[]
        for i in range(len(path)-1):
            weight_gaps.append(  rng.normal(1,.36)* G[path[i]][path[i+1]][weight])
            dists.append( G[path[i]][path[i+1]]['minDist'])# np.linalg.norm(newCoords[i]-newCoords[i+1]))
            
        weight_gaps = np.array(weight_gaps)
        dists = np.array(dists)
        
        sum_weight = np.sum(weight_gaps)
        ave_weight =np.mean(weight_gaps)
        
         
        #ACS Omega paper page G
        pathInfo ={ 'path':path,   'dists':dists}
        pathInfo[weight]=sum_weight
        pathInfo[weight +"s" ]=weight_gaps
        pathInfo['ave' + weight ]=ave_weight
        shortestPaths.append(pathInfo )
    
    return shortestPaths

def MinDistanceA(donor,accepter):
    """Walks through what should be the residue and finds the closest distance between the two redox cofactors

    Args:
        donor (atom): electron source
        accepter (atom): electron sink

    Returns:
        float: distance in Angstroms between the two redox cofactors
    """
    r_min = 1000

    for D in donor['redoxPoints']:
        for A in accepter['redoxPoints']:
            d = np.linalg.norm(A[1] - D[1])
            if d < r_min:
                # minimum distance
                r_min = d

    return  r_min*10  # convert to angstroms
    
    
def DistanceRates(atom_COM,r_min,donor, acceptor, attemptFrequency, vibrationRadius_nm,dutton_radius_A, r_COM_A):
    """Calculate the distance effect on electron transfer.  

    Args:
        protDensity (float): density of the protein media between the donor and acceptor
        r_min (float): distance between the donor and acceptor

    Returns:
        tuple<float>: distance_rate_static, distance_rate_vibrate, distance_rate_min
    """
    
    # determine the occupancy of the path between the two redox cofactors
    protDensity =  getNeighborOccupancy(donor, acceptor, atom_COM)
                
    # protDensity is going to vary between 0 and 1.  -.9 represents tunneling through the 100% protein media, while -2.8 estimates tunneling through vacuum or water
    #water rate is taken from Long-range electron transferHarry B. Gray†and Jay R. Winkler
    gamma =    -1.2*protDensity - 1.6*(1-protDensity)
    

    # distance rate for fixed pdb file coordinates
    distance_rate_static = attemptFrequency * np.exp(gamma * r_min)
    # distance adjustment for vibrating pdb file coordinates  
    vibrate = vibrationRadius_nm*10.0
     
    
    rv_min = r_min - vibrate
    alter=100
    if  donor['std'] > 0 and acceptor['std'] > 0:
        vibrate = np.mean([donor['std'], acceptor['std']])*10 #convert to angstroms
        rv_min = r_min - vibrate
        alter =5* np.exp(gamma * gamma * vibrate) 
    if rv_min < 0:
        rv_min = .1
    
        
    distance_rate_vibrate = attemptFrequency*(np.exp(gamma * (r_min-vibrate)) + np.exp(
        gamma * r_min) + np.exp(gamma * (r_min+vibrate)))/3

    # cheat for g value by using the dutton radius to form a sharp transition between the distant dependant reqime and the distant independent regime
    # simplification of figure 6 from Electron Tunneling in Biology: When Does it Matter?
    
    if r_COM_A < dutton_radius_A :
        distance_rate_min = attemptFrequency*np.exp(-1.4 * dutton_radius_A) *alter
    else:
#        distance_rate_min = attemptFrequency*np.exp(gamma * (r_min))#*1+ 4*r_COM_A)/5)
        distance_rate_min = attemptFrequency*np.exp(-1.4  * (r_COM_A))*alter#*1+ 4*r_COM_A)/5)

    return  gamma, distance_rate_static, distance_rate_vibrate, distance_rate_min

def RedoxRates(donor, acceptor,reorgE_EV, beta):
    """Calculate the electron transfer rates between two redox cofactors, use the standard potentials to estimate delta G0
    voltage differences between the factors are added into the delta G0

    Args:
        donor (atom): electron source
        acceptor (atom): electron sink
        reorgE_EV (float): _description_
        beta (float): _description_
    """
    #get the energy difference between the two redox cofactors    
    dE = acceptor['redoxEnergy_EV']-  donor['redoxEnergy_EV'] 
    dV = acceptor['totalPotential'] - donor['totalPotential'] 
    
    
    #get the forward rate 
    dF_F = (reorgE_EV + dE+dV)**2/(4*reorgE_EV)
    energy_rate_forward = np.exp(-beta * dF_F)
    
    #get the backard rate for returning to acceptor
    dF_B = (reorgE_EV - dE -dV)**2/(4*reorgE_EV)
    energy_rate_back = np.exp(-beta * dF_B)
    
    return dE+dV,dF_F,dF_B,energy_rate_forward,energy_rate_back

    
def RateNetwork(G):
    """Precompute all the rates for leaving this node to make KMC faster and help some of the graphing
       Get the rates for leaving this node and then sort them to allow a quick lookup of the next node

    Args:
        G (graph): graph that has the rates assigned to the edges 
    """
    for node in G.nodes:
            
        rates = []
        
        otherNode =[]
        for edge in G.out_edges(node):
            rates.append( G.edges[edge]['rate'])
            otherNode.append(edge[1])
        rates = np.array(rates)
        sort=np.argsort(-1*rates)
        rates = np.cumsum(rates[sort])
        max = rates[-1]
        rates = rates/max
        targets = [otherNode[x] for x in sort]
        
    
        G.nodes[node]['rates'] = rates #cumulative probability of leaving this node
        G.nodes[node]['targets'] = targets #nodes assigned to each rate
        G.nodes[node]['outtime'] =1/ max  #average time to leave this node in ns
    
def MoveElectron(G_test, electronLocation):
    """Walk through the probabilities avaible at this node and select the node based on a random number
    The nodes have been arranged from most probably to least to allow a quick lookup

    Args:
        G_test (graph): graph that has the exit probabilities for each node available
        electronLocation ( int ): current node location of electron

    Returns:
        _type_: next node location,                time to move to next node
    """
    newLocation=-1
    #get all the options for electron at this step
    rates = G_test.nodes[electronLocation]['rates']
    
    v=rng.random()
    for i in range(len(rates)):
        if v<rates[i]:
            newLocation = G_test.nodes[electronLocation]['targets'][i]
            break
    if newLocation == -1:
        newLocation = G_test.nodes[electronLocation]['targets'][-1]
    timeStep = G_test.nodes[newLocation]['outtime'] 
    
    return newLocation, timeStep

def KMC(G_test,activeAminos, injectionAminos,exitAminos, numberElectrons=5000, maxIterations = 500000 ):
    """_summary_

    Args:
        G_test (_type_): _description_
        startElectronNode (_type_): _description_
        activeAminos (_type_): _description_
        endElectroneNode (_type_): _description_
        numberElectrons (int, optional): _description_. Defaults to 5000.
        maxIterations (int, optional): _description_. Defaults to 500000.

    Returns:
        _type_: _description_
    """
    
    startElectronNodes=[amino['aminoIndex'] for amino in injectionAminos]
    endElectroneNodes= [amino['aminoIndex'] for amino in exitAminos]
        
    successDwellTimes = np.zeros(len(activeAminos))
    dwellTimes = np.zeros(len(activeAminos))
    passes = np.zeros(len(activeAminos))
    electronTimes =[]
    diffusions =[]
    
    locations = np.zeros(maxIterations, dtype=int)
    distances = np.zeros(maxIterations, dtype=float)
    times = np.zeros(maxIterations, dtype=float)
    
    for attempt in range(numberElectrons):
         
        #choose a random start and end point
        electronLocation=  startElectronNodes[rng.integers(0,len(startElectronNodes))]
        seeking = True
        cc=-1
         
        while seeking:
            cc+=1
            
            #look at the rates to get the next location
            newLocation, Q=MoveElectron(G_test, electronLocation)
            
            #move the time forward based on the rates available
            timeStep = 1/Q*np.log(1/rng.random())
            
            #get the distance that will be jumped
            dx=G_test[electronLocation][newLocation]['minDist']
            #dx= np.linalg.norm( activeAminos[electronLocation]['centerOfMass']-activeAminos[newLocation]['centerOfMass'])
          
            #record the locations and rates
            locations[cc] = electronLocation
            distances[cc] =  dx
            times[cc] = timeStep
            
            #mark the current location with how long the electron stays there
            dwellTimes[electronLocation] += timeStep
            #mark that the electron has passed through this location
            passes[electronLocation] += 1
            
            #move the electron to the next location
            electronLocation = newLocation
             
            #if the electron has reached the injection node, record the dwell times
            if (cc>=maxIterations-1):
                seeking = False
                diffusions.append(np.mean( (distances[:cc]**2)/(2*times[:cc])))
                cc=-1  
            #check if we have found the endpoint
            elif electronLocation in endElectroneNodes:
                seeking = False
                for i in range(cc):
                    successDwellTimes[locations[i]] += times[i]
                 
                diffusions.append(np.mean( (distances[:cc]**2)/(2*times[:cc])))
                electronTimes.append(np.sum( times[:cc]))
                cc=-1
   
    return successDwellTimes, dwellTimes, passes, electronTimes, diffusions     


def ConnectGraphs(activeAminos, atom_COM,reorgE_EV, beta,maxInteraction_radius_nm,attemptFrequency, vibrationRadius_nm,dutton_radius_nm, verbose=False):
    for i in range(len(activeAminos)):
        activeAminos[i]['neighbors'] = [] #clear out the last list to make debugging easier
        
        
    G_static = nx.DiGraph()
    G_vibrate = nx.DiGraph()
    G_min = nx.DiGraph()
    G_connected = nx.Graph()
    
    for i in range(len(activeAminos)):
        G_static.add_node(i, label=f"{activeAminos[i]['amino']} {activeAminos[i]['index']}")
        G_vibrate.add_node(i, label=f"{activeAminos[i]['amino']} {activeAminos[i]['index']}")
        G_min .add_node(i, label=f"{activeAminos[i]['amino']} {activeAminos[i]['index']}")
        G_connected .add_node(i, label=f"{activeAminos[i]['amino']} {activeAminos[i]['index']}")
    
    distanceRates = []
    energyRates = []
    transferRates = []
    voltageRates = []
    gammas = []

    # determine the pairwise distance between each redox cofactor
    for i in range(len(activeAminos)):
        for j in range(i+1, len(activeAminos)):
            r_COM = np.linalg.norm(activeAminos[i]['centerOfMass']-activeAminos[j]['centerOfMass'])

            # check if the redox cofactors are within the interaction radius

            if r_COM < maxInteraction_radius_nm:
                 
                # walk through all the redox cofactor atoms and determine the minimum distance to the other redox cofactor atoms
                r_min =  MinDistanceA(activeAminos[i], activeAminos[j])

                ##################################################################################
                #################     Determine distance prefactor Vr ############################
                ##################################################################################
                
                #get the rates for the different distance regimes
                gamma, distance_rate_static, distance_rate_vibrate, distance_rate_min= DistanceRates(atom_COM, r_min,activeAminos[i], activeAminos[j],
                                                                                                     attemptFrequency, vibrationRadius_nm,dutton_radius_nm*10, r_COM*10)
                
                #store for graphing
                gammas.append(gamma)
                distanceRates.append(  [r_min, distance_rate_static, distance_rate_vibrate, distance_rate_min])
                
                ##################################################################################
                #################     Determine energy rates          ############################
                ##################################################################################
                # determine the energy difference between the two redox cofactors
                dE,dF_F,dF_B,energy_rate_forward,energy_rate_back=RedoxRates(activeAminos[i], activeAminos[j],reorgE_EV, beta)
                
                #energy_rate_forward=1
                #energy_rate_back=1

                k_NA_forward = [distance_rate_static*energy_rate_forward, distance_rate_vibrate * energy_rate_forward, distance_rate_min*energy_rate_forward]
                k_NA_back    = [distance_rate_static*energy_rate_back,    distance_rate_vibrate * energy_rate_back,    distance_rate_min*energy_rate_back]
                
                #store for graphing: give examples of the transitions for 1nm gap at average protein density
                energyRates.append([dE, attemptFrequency*np.exp(-14)* energy_rate_forward,attemptFrequency*np.exp(-14)*energy_rate_back]) #include the attempt frequency and rate at 1 nm for 1.4 1/nm gamma
                transferRates.append([r_min,  k_NA_forward[0] ,   k_NA_forward[1] ,   k_NA_forward[2], k_NA_back[2]  , r_COM ])
                
                ##################################################################################
                #################     book keeping for networks       ############################
                ##################################################################################
                
                G_static.add_edge(i, j, time=1e9/k_NA_forward[0] , rate = k_NA_forward[0]*1e-9, dist = r_COM, minDist = r_min/10.0 )
                G_static.add_edge(j, i, time=1e9/k_NA_back[0] ,rate = k_NA_back[0]*1e-9, dist = r_COM, minDist = r_min/10.0 )
                
                G_vibrate.add_edge(i, j, time=1e9/k_NA_forward[1], rate = k_NA_forward[1]*1e-9, dist = r_COM, minDist = r_min /10.0)
                G_vibrate.add_edge(j, i, time=1e9/k_NA_back[1], rate = k_NA_back[1]*1e-9, dist = r_COM, minDist = r_min/10.0 )
                
                G_min.add_edge(i, j, time=1e9/k_NA_forward[2] ,rate = k_NA_forward[2]*1e-9, dist = r_COM, minDist = r_min /10.0)
                G_min.add_edge(j, i, time=1e9/k_NA_back[2] ,rate = k_NA_back[2]*1e-9, dist = r_COM, minDist = r_min/10.0 )
                
                G_connected.add_edge(i, j, time=1e9/k_NA_forward[0] + 1e9/k_NA_back[0], distance = r_COM, minDist = r_min/10.0  )
             
    if verbose:
        # determine the connectedness of the redox cofactors
        degrees = sorted(d for n, d in G_vibrate.degree())
        print(f"min neighbors: {np.min(degrees)}")
        print(f"max neighbors: {np.max(degrees)}")
        print(f"mean neighbors: {np.mean(degrees)}")
        print(f"scale metric of graph: {nx.s_metric(G_vibrate, normalized=False)}")
        numberLocalBridges = 0
        for bridge in nx.local_bridges(G_connected):
            numberLocalBridges += 1
        print(f"local bridges: {numberLocalBridges}")
        print(f"std neighbors: {np.std(degrees)}")
        print(f'number of redox cofactors: { G_vibrate.number_of_nodes()}')
        print(f'number of tunnel gaps: { G_vibrate.number_of_edges()}')        
    
        
    return G_static, G_vibrate, G_min, G_connected, (gammas, distanceRates, energyRates, transferRates, voltageRates)