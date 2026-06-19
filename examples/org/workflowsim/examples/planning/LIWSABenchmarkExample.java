/**
 * Copyright 2012-2013 University Of Southern California
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
package org.workflowsim.examples.planning;

import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;
import org.cloudbus.cloudsim.CloudletSchedulerSpaceShared;
import org.cloudbus.cloudsim.DatacenterCharacteristics;
import org.cloudbus.cloudsim.HarddriveStorage;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.VmAllocationPolicySimple;
import org.cloudbus.cloudsim.VmSchedulerTimeShared;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import org.workflowsim.CondorVM;
import org.workflowsim.Job;
import org.workflowsim.WorkflowDatacenter;
import org.workflowsim.WorkflowEngine;
import org.workflowsim.WorkflowPlanner;
import org.workflowsim.utils.ClusteringParameters;
import org.workflowsim.utils.OverheadParameters;
import org.workflowsim.utils.Parameters;
import org.workflowsim.utils.ReplicaCatalog;

/**
 * Benchmark driver: runs HEFT, Min-Min, MLEAO, and LIWSA on the same DAX
 * file and prints a side-by-side comparison table suitable for inclusion in
 * a research paper.
 *
 * Each algorithm is run as a complete, isolated WorkflowSim simulation
 * (CloudSim.init() -> startSimulation() -> stopSimulation()), so global
 * CloudSim state is fully reset between runs.
 *
 * All configuration is in the CONFIGURATION block at the top of main().
 *
 * Prerequisites (same as LIWSAPlanningAlgorithmExample):
 *   1. LIWSA and MLEAO added to Parameters.PlanningAlgorithm enum
 *   2. LIWSA and MLEAO cases added to WorkflowPlanner.getPlanningAlgorithm()
 *   3. LIWSAPlanningAlgorithm.java and MLEAOPlanningAlgorithm.java in
 *      sources/org/workflowsim/planning/
 */
public class LIWSABenchmarkExample {

    // ---------------------------------------------------------------
    // VM type table.  { mips, bandwidthMbitS, costPerSec, ram_MB,
    //                   storage_MB, count }
    // ---------------------------------------------------------------
    private static final double[][] VM_TYPES = {
        { 250.0, 160.0, 0.15, 512, 10000, 4},  // Micro  x4
        { 500.0, 160.0, 0.30, 512, 10000, 4},  // Small  x4
        {1000.0, 160.0, 0.60, 512, 10000, 4},  // Medium x4
        {2000.0, 160.0, 0.90, 512, 10000, 4},  // Large  x4
    };

    /** Result container for one algorithm run. */
    private static class RunResult {
        String name;
        double makespan;
        double cost;
        double wallClockMs;
        int jobCount;
    }

    public static void main(String[] args) {

        // ==============================================================
        // CONFIGURATION
        // ==============================================================

        // Workflow DAX files to evaluate. The benchmark loops over all of
        // them and prints one result row per (algorithm, workflow) pair.
        // Adjust paths to match your environment.
        String[] daxFiles = {
            "config/dax/Montage_50.xml",
            "config/dax/Montage_100.xml",
            "config/dax/CyberShake_30.xml",
            "config/dax/CyberShake_50.xml",
            "config/dax/SIPHT_60.xml",
        };

        // LIWSA and MLEAO shared parameters
        int  populationSize  = 30;
        int  generationCount = 100;
        long randomSeed      = 7L;    // fixed for reproducibility; change to 0 for random

        // ==============================================================
        // END CONFIGURATION
        // ==============================================================

        DecimalFormat df2 = new DecimalFormat("######.##");
        DecimalFormat df0 = new DecimalFormat("######");

        System.out.println();
        System.out.printf("%-25s %-10s %10s %10s %10s%n",
            "Workflow", "Algorithm", "Makespan(s)", "Cost", "Wall(ms)");
        System.out.println("-".repeat(70));

        for (String daxPath : daxFiles) {
            if (!new File(daxPath).exists()) {
                System.out.println("Skipping (not found): " + daxPath);
                continue;
            }
            String workflowName = new File(daxPath).getName().replace(".xml", "");

            RunResult heft   = runPlanning(daxPath, Parameters.PlanningAlgorithm.HEFT,
                                           Parameters.SchedulingAlgorithm.STATIC,
                                           "HEFT",     populationSize, generationCount, randomSeed);
            RunResult mleao  = runPlanning(daxPath, Parameters.PlanningAlgorithm.MLEAO,
                                           Parameters.SchedulingAlgorithm.STATIC,
                                           "MLEAO",    populationSize, generationCount, randomSeed);
            RunResult liwsa  = runPlanning(daxPath, Parameters.PlanningAlgorithm.LIWSA,
                                           Parameters.SchedulingAlgorithm.STATIC,
                                           "LIWSA",    populationSize, generationCount, randomSeed);
            RunResult liwsaML = runPlanning(daxPath, Parameters.PlanningAlgorithm.LIWSAML,
                                            Parameters.SchedulingAlgorithm.STATIC,
                                            "LIWSA-ML", populationSize, generationCount, randomSeed);
            RunResult minmin = runPlanning(daxPath, Parameters.PlanningAlgorithm.INVALID,
                                           Parameters.SchedulingAlgorithm.MINMIN,
                                           "Min-Min",  populationSize, generationCount, randomSeed);

            for (RunResult r : new RunResult[]{heft, minmin, mleao, liwsa, liwsaML}) {
                if (r == null) continue;
                System.out.printf("%-25s %-10s %10s %10s %10s%n",
                    workflowName, r.name,
                    df2.format(r.makespan),
                    df2.format(r.cost),
                    df0.format(r.wallClockMs));
            }
            System.out.println();

            // Print improvement ratios with HEFT as the baseline
            if (heft != null && liwsa != null && mleao != null) {
                System.out.printf("  [%s] LIWSA    vs HEFT:    makespan %+.1f%%  cost %+.1f%%%n",
                    workflowName,
                    100.0 * (liwsa.makespan - heft.makespan) / heft.makespan,
                    100.0 * (liwsa.cost - heft.cost) / heft.cost);
                System.out.printf("  [%s] LIWSA-ML vs HEFT:    makespan %+.1f%%  cost %+.1f%%%n",
                    workflowName,
                    liwsaML != null ? 100.0*(liwsaML.makespan-heft.makespan)/heft.makespan : Double.NaN,
                    liwsaML != null ? 100.0*(liwsaML.cost-heft.cost)/heft.cost             : Double.NaN);
                System.out.printf("  [%s] LIWSA-ML vs MLEAO:   makespan %+.1f%%  cost %+.1f%%%n",
                    workflowName,
                    liwsaML != null ? 100.0*(liwsaML.makespan-mleao.makespan)/mleao.makespan : Double.NaN,
                    liwsaML != null ? 100.0*(liwsaML.cost-mleao.cost)/mleao.cost             : Double.NaN);
                System.out.printf("  [%s] LIWSA-ML vs LIWSA:   makespan %+.1f%%  cost %+.1f%%%n",
                    workflowName,
                    liwsaML != null ? 100.0*(liwsaML.makespan-liwsa.makespan)/liwsa.makespan : Double.NaN,
                    liwsaML != null ? 100.0*(liwsaML.cost-liwsa.cost)/liwsa.cost             : Double.NaN);
                System.out.println();
            }
        }
    }

    /**
     * Runs one complete WorkflowSim simulation and returns the result.
     * CloudSim state is fully re-initialised each call, so runs are
     * independent regardless of order.
     */
    private static RunResult runPlanning(
            String daxPath,
            Parameters.PlanningAlgorithm planningAlg,
            Parameters.SchedulingAlgorithm schedulingAlg,
            String label,
            int popSize, int genCount, long seed) {

        try {
            int totalVMs = 0;
            for (double[] t : VM_TYPES) { totalVMs += (int) t[5]; }

            OverheadParameters op = new OverheadParameters(0, null, null, null, null, 0);
            ClusteringParameters cp = new ClusteringParameters(
                0, 0, ClusteringParameters.ClusteringMethod.NONE, null
            );

            Parameters.init(totalVMs, daxPath, null, null, op, cp,
                            schedulingAlg, planningAlg, null, 0);
            ReplicaCatalog.init(ReplicaCatalog.FileSystem.LOCAL);
            CloudSim.init(1, Calendar.getInstance(), false);

            WorkflowDatacenter dc = createDatacenter("DC_" + label);
            WorkflowPlanner wfPlanner = new WorkflowPlanner("planner_" + label, 1);
            WorkflowEngine wfEngine = wfPlanner.getWorkflowEngine();
            List<CondorVM> vmList = createVMs(wfEngine.getSchedulerId(0));
            wfEngine.submitVmList(vmList, 0);
            wfEngine.bindSchedulerDatacenter(dc.getId(), 0);

            long t0 = System.currentTimeMillis();
            CloudSim.startSimulation();
            List<Job> jobs = wfEngine.getJobsReceivedList();
            CloudSim.stopSimulation();
            long wall = System.currentTimeMillis() - t0;

            RunResult r = new RunResult();
            r.name = label;
            r.wallClockMs = wall;
            r.jobCount = jobs.size();
            for (Job job : jobs) {
                if (job.getClassType() ==
                        org.workflowsim.utils.Parameters.ClassType.STAGE_IN.value) {
                    continue;
                }
                r.makespan = Math.max(r.makespan, job.getFinishTime());
                r.cost += job.getActualCPUTime() * job.getCostPerSec();
            }
            return r;

        } catch (Exception e) {
            System.err.println("[" + label + "] simulation failed: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    private static List<CondorVM> createVMs(int userId) {
        LinkedList<CondorVM> list = new LinkedList<>();
        int vmId = 0;
        for (double[] type : VM_TYPES) {
            double mips = type[0]; long bw = (long) type[1];
            double cost = type[2]; int ram = (int) type[3];
            long storage = (long) type[4]; int count = (int) type[5];
            for (int k = 0; k < count; k++) {
                list.add(new CondorVM(vmId++, userId, mips, 1, ram, bw, storage,
                    "Xen", cost, 0.0, 0.0, 0.0, new CloudletSchedulerSpaceShared()));
            }
        }
        return list;
    }

    private static WorkflowDatacenter createDatacenter(String name) {
        int totalVMs = 0;
        for (double[] t : VM_TYPES) { totalVMs += (int) t[5]; }
        int numHosts = Math.max(1, (totalVMs + 3) / 4);

        List<Host> hostList = new ArrayList<>();
        for (int h = 0; h < numHosts; h++) {
            List<Pe> peList = new ArrayList<>();
            for (int p = 0; p < 16; p++) {
                peList.add(new Pe(p, new PeProvisionerSimple(2000)));
            }
            hostList.add(new Host(h,
                new RamProvisionerSimple(8192),
                new BwProvisionerSimple(100000),
                1000000L, peList,
                new VmSchedulerTimeShared(peList)));
        }

        DatacenterCharacteristics dc = new DatacenterCharacteristics(
            "x86", "Linux", "Xen", hostList, 10.0, 3.0, 0.05, 0.1, 0.1);
        WorkflowDatacenter datacenter = null;
        try {
            HarddriveStorage s = new HarddriveStorage(name, 1e12);
            s.setMaxTransferRate(100);
            LinkedList<Storage> sl = new LinkedList<>();
            sl.add(s);
            datacenter = new WorkflowDatacenter(name, dc,
                new VmAllocationPolicySimple(hostList), sl, 0);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return datacenter;
    }
}
