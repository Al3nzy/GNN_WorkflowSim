package org.workflowsim.examples.planning;

import java.io.File;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;
import org.cloudbus.cloudsim.CloudletSchedulerSpaceShared;
import org.cloudbus.cloudsim.DatacenterCharacteristics;
import org.cloudbus.cloudsim.HarddriveStorage;
import org.cloudbus.cloudsim.Host;
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
import org.workflowsim.planning.LIWSAGNNPlanningAlgorithm;
import org.workflowsim.utils.ClusteringParameters;
import org.workflowsim.utils.OverheadParameters;
import org.workflowsim.utils.Parameters;
import org.workflowsim.utils.ReplicaCatalog;

/**
 * Benchmark demonstration for LIWSA-GNN scheduling algorithm.
 * Evaluates all workflow families (Montage, CyberShake, Epigenomics, Inspiral, Sipht)
 * across Seeds 1 to 5 and exports results to results/gnn_benchmark_results.csv.
 */
public class LIWSAGNNBenchmarkExample {

    private static final double[][] VM_TYPES = {
        { 250.0, 160.0, 0.15, 512, 10000, 4},  // Micro  x4
        { 500.0, 160.0, 0.30, 512, 10000, 4},  // Small  x4
        {1000.0, 160.0, 0.60, 512, 10000, 4},  // Medium x4
        {2000.0, 160.0, 0.90, 512, 10000, 4},  // Large  x4
    };

    public static void main(String[] args) {
        try {
            // Declare the String array properly instead of returning it
            String[] daxFiles = new String[] {
                // Montage
                "config/dax/Montage_25.xml",
                "config/dax/Montage_50.xml",
                "config/dax/Montage_100.xml",
                "config/dax/Montage_1000.xml",      // 1,000 tasks

                // CyberShake
                "config/dax/CyberShake_30.xml",
                "config/dax/CyberShake_50.xml",
                "config/dax/CyberShake_100.xml",
                "config/dax/CyberShake_1000.xml",   // 1,000 tasks

                // Epigenomics
                "config/dax/Epigenomics_24.xml",
                "config/dax/Epigenomics_46.xml",
                "config/dax/Epigenomics_100.xml",
                "config/dax/Epigenomics_997.xml",  // 997 tasks (use Epigenomics_997.xml if renamed)

                // Inspiral
                "config/dax/Inspiral_30.xml",
                "config/dax/Inspiral_50.xml",
                "config/dax/Inspiral_100.xml",
                "config/dax/Inspiral_1000.xml",     // 1,000 tasks

                // Sipht
                "config/dax/Sipht_30.xml",
                "config/dax/Sipht_60.xml",
                "config/dax/Sipht_100.xml",
                "config/dax/Sipht_1000.xml"        // 1,000 tasks
            };

            int numSeeds = 5;

            // Ensure results output directory exists
            File resultsDir = new File("results");
            if (!resultsDir.exists()) {
                resultsDir.mkdirs();
            }

            System.out.println("=".repeat(70));
            System.out.println("LIWSA-GNN Multi-Workflow Benchmark (5 Seeds x All Workflows)");
            System.out.println("=".repeat(70));
            System.out.println("Seeds: 1 to " + numSeeds);
            System.out.println("Algorithm: LIWSA-GNN (GNN-guided warm-start)");
            System.out.println();

            long benchmarkStart = System.currentTimeMillis();

            for (String daxFile : daxFiles) {
                File file = new File(daxFile);
                if (!file.exists()) {
                    System.err.println("Skipping missing DAX file: " + daxFile);
                    continue;
                }

                System.out.println(">>> Running Workflow: " + file.getName());
                for (int seed = 1; seed <= numSeeds; seed++) {
                    System.out.println("--- Seed " + seed + " ---");
                    runGNNPlanningExample(daxFile, seed);
                }
                System.out.println();
            }

            long totalWall = System.currentTimeMillis() - benchmarkStart;
            System.out.println("=".repeat(70));
            System.out.printf("FULL BENCHMARK COMPLETED: %.1f seconds%n", totalWall / 1000.0);
            System.out.println("Results written to: results/gnn_benchmark_results.csv");
            System.out.println("=".repeat(70));

        } catch (Exception e) {
            System.err.println("ERROR: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Run a single planning example with LIWSA-GNN
     */
    private static void runGNNPlanningExample(String daxFile, long seed) {
        try {
            LIWSAGNNPlanningAlgorithm.CONFIG_WEIGHTS_PATH = "gnn_weights.txt";
            LIWSAGNNPlanningAlgorithm.CONFIG_NUM_CANDIDATES = 80;
            LIWSAGNNPlanningAlgorithm.CONFIG_NUM_SEEDS_FROM_GNN = 4;

            int totalVMs = 0;
            for (double[] t : VM_TYPES) { totalVMs += (int) t[5]; }

            // Setup parameters
            OverheadParameters op = new OverheadParameters(0, null, null, null, null, 0);
            ClusteringParameters cp = new ClusteringParameters(
                0, 0, ClusteringParameters.ClusteringMethod.NONE, null);

            // Setup parameters with LIWSAGNN planning algorithm
            Parameters.init(totalVMs, daxFile, null, null, op, cp,
                Parameters.SchedulingAlgorithm.STATIC, Parameters.PlanningAlgorithm.LIWSAGNN, null, 0);
            Parameters.setCostModel(Parameters.CostModel.VM);
            LIWSAGNNPlanningAlgorithm.CONFIG_RANDOM_SEED = seed;

            ReplicaCatalog.init(ReplicaCatalog.FileSystem.LOCAL);
            
            // Initialize CloudSim with 1 user
            CloudSim.init(1, Calendar.getInstance(), false);

            // Create infrastructure
            WorkflowDatacenter dc = createDatacenter("DC_GNN_" + seed);
            WorkflowPlanner wfPlanner = new WorkflowPlanner("planner_GNN_" + seed, 1);
            WorkflowEngine wfEngine = wfPlanner.getWorkflowEngine();
            List<CondorVM> vmList = createVMs(wfEngine.getSchedulerId(0));
            wfEngine.submitVmList(vmList, 0);
            wfEngine.bindSchedulerDatacenter(dc.getId(), 0);

            // Run simulation
            long simStart = System.currentTimeMillis();
            CloudSim.startSimulation();
            List<Job> jobs = wfEngine.getJobsReceivedList();
            CloudSim.stopSimulation();
            long simWall = System.currentTimeMillis() - simStart;

            // Compute metrics
            double makespan = 0;
            double cost = 0;
            int completedJobs = 0;
            
            for (Job job : jobs) {
                makespan = Math.max(makespan, job.getFinishTime());
                cost += job.getProcessingCost();
                completedJobs++;
            }
            double simTimeSec = simWall / 1000.0;
            String daxName = new File(daxFile).getName();

            System.out.printf("  Completed jobs: %d%n", completedJobs);
            System.out.printf("  Makespan: %.2f seconds%n", makespan);
            System.out.printf("  Total cost: %.2f%n", cost);
            System.out.printf("  Simulation time: %.1f seconds%n", simTimeSec);

            // Save results to CSV with automatic header creation
            File csvFile = new File("results/gnn_benchmark_results.csv");
            boolean isNewFile = !csvFile.exists() || csvFile.length() == 0;

            try (java.io.FileWriter fw = new java.io.FileWriter(csvFile, true)) {
                if (isNewFile) {
                    fw.write("Workflow_DAX,Algorithm,Seed,Makespan_s,Cost,Sim_Time_s\n");
                }
                fw.write(String.format("%s,LIWSA-GNN,%d,%.4f,%.4f,%.2f%n", daxName, seed, makespan, cost, simTimeSec));
            } catch (java.io.IOException e) {
                System.err.println("Failed to write CSV: " + e.getMessage());
            }

        } catch (Exception e) {
            System.err.println("  ERROR: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Create virtual machines for the simulation
     */
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

    /**
     * Create a workflow datacenter
     */
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