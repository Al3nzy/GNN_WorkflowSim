package org.workflowsim.examples.planning;

import java.io.File;
import java.text.DecimalFormat;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.CloudletSchedulerSpaceShared;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.core.CloudSim;
import org.workflowsim.CondorVM;
import org.workflowsim.Job;
import org.workflowsim.WorkflowDatacenter;
import org.workflowsim.WorkflowEngine;
import org.workflowsim.WorkflowPlanner;
import org.workflowsim.examples.WorkflowSimBasicExample1;
import org.workflowsim.utils.ClusteringParameters;
import org.workflowsim.utils.OverheadParameters;
import org.workflowsim.utils.Parameters;
import org.workflowsim.utils.ReplicaCatalog;

/**
 * Run WorkflowSim with HEFT planning algorithm and compute metrics.
 */
public class HEFTBenchmark extends WorkflowSimBasicExample1 {

    private static final double COST_PER_SECOND = 3.0; // match Locust

    ////////////////////////// STATIC METHODS ///////////////////////
    protected static List<CondorVM> createVM(int userId, int vms) {
        LinkedList<CondorVM> list = new LinkedList<>();

        // VM Parameters (NOTE: keep same for Locust!)
        long size = 1000; // image size (MB)
        int ram = 512; // vm memory (MB)
        int mips = 1000;
        long bw = 1000;
        int pesNumber = 1;
        String vmm = "Xen";

        CondorVM[] vm = new CondorVM[vms];
        Random bwRandom = new Random(System.currentTimeMillis());
//        for (int i = 0; i < vms; i++) {
//            double ratio = bwRandom.nextDouble();
//            vm[i] = new CondorVM(i, userId, (int)(mips * ratio), pesNumber, ram,
//                    (long) (bw * ratio), size, vmm, new CloudletSchedulerSpaceShared());
//            list.add(vm[i]);
//        }
        for (int i = 0; i < vms; i++) {
           // double ratio = bwRandom.nextDouble();
            vm[i] = new CondorVM(i, userId, mips, pesNumber, ram,
                    bw , size, vmm, new CloudletSchedulerSpaceShared());
            list.add(vm[i]);
        }
        return list;
    }

    public static void main(String[] args) {
        try {
            int vmNum = 5; // must also be used in Locust for fair comparison
            String daxPath = "examples/org/workflowsim/examples/workflowDatasets/Montage_50.xml";

            File daxFile = new File(daxPath);
            if (!daxFile.exists()) {
                Log.printLine("Warning: Please replace daxPath with the physical path in your working environment!");
                return;
            }

            Parameters.SchedulingAlgorithm sch_method = Parameters.SchedulingAlgorithm.STATIC;
            Parameters.PlanningAlgorithm pln_method = Parameters.PlanningAlgorithm.HEFT;
            ReplicaCatalog.FileSystem file_system = ReplicaCatalog.FileSystem.LOCAL;

            OverheadParameters op = new OverheadParameters(0, null, null, null, null, 0);
            ClusteringParameters.ClusteringMethod method = ClusteringParameters.ClusteringMethod.NONE;
            ClusteringParameters cp = new ClusteringParameters(0, 0, method, null);

            Parameters.init(vmNum, daxPath, null, null, op, cp, sch_method, pln_method, null, 0);
            ReplicaCatalog.init(file_system);

            CloudSim.init(1, Calendar.getInstance(), false);

            WorkflowDatacenter datacenter0 = createDatacenter("Datacenter_0");

            WorkflowPlanner wfPlanner = new WorkflowPlanner("planner_0", 1);
            WorkflowEngine wfEngine = wfPlanner.getWorkflowEngine();

            List<CondorVM> vmlist0 = createVM(wfEngine.getSchedulerId(0), Parameters.getVmNum());
            wfEngine.submitVmList(vmlist0, 0);
            wfEngine.bindSchedulerDatacenter(datacenter0.getId(), 0);

            CloudSim.startSimulation();
            List<Job> outputList0 = wfEngine.getJobsReceivedList();
            CloudSim.stopSimulation();

            // Print default job list (from WorkflowSimBasicExample1)
            printJobList(outputList0);

            // Compute additional metrics
            computeMetrics(outputList0, vmNum);

        } catch (Exception e) {
            Log.printLine("The simulation has been terminated due to an unexpected error");
            e.printStackTrace();
        }
    }

    /** Compute makespan, cost, utilization, throughput, avg response, waiting time, load balance */
    private static void computeMetrics(List<Job> jobs, int vmNum) {
        double makespan = 0;
        double totalCpu = 0;
        double totalResponseTime = 0;
        double totalWaitingTime = 0;

        int jobCount = jobs.size();

        // Track per-VM load for load balancing metric
        double[] vmLoad = new double[vmNum];

        for (Job job : jobs) {
            makespan = Math.max(makespan, job.getFinishTime());
            totalCpu += job.getActualCPUTime();

            double response = job.getFinishTime() - job.getExecStartTime();
            double waiting = job.getExecStartTime(); // since submission ~0

            totalResponseTime += response;
            totalWaitingTime += waiting;

            if (job.getVmId() >= 0 && job.getVmId() < vmNum) {
                vmLoad[job.getVmId()] += job.getActualCPUTime();
            }
        }

        double cost = totalCpu * COST_PER_SECOND;
        double utilization = totalCpu / (makespan * vmNum);
        double throughput = jobCount / makespan;
        double avgResponse = totalResponseTime / jobCount;
        double avgWaiting = totalWaitingTime / jobCount;

        // Load balance = std deviation of VM loads
        double avgLoad = totalCpu / vmNum;
        double sumSq = 0;
        for (double load : vmLoad) {
            sumSq += Math.pow(load - avgLoad, 2);
        }
        double loadBalance = Math.sqrt(sumSq / vmNum);

        DecimalFormat df = new DecimalFormat("###.###");
        System.out.println("---------- METRICS (HEFT) ----------");
        System.out.println("Makespan       = " + df.format(makespan));
        System.out.println("Cost           = " + df.format(cost));
        System.out.println("Utilization    = " + df.format(utilization));
        System.out.println("Throughput     = " + df.format(throughput));
        System.out.println("Avg Response   = " + df.format(avgResponse));
        System.out.println("Avg Waiting    = " + df.format(avgWaiting));
        System.out.println("Load Balanceσ  = " + df.format(loadBalance));
    }
}

//---------- FINAL METRICS (Locust) ----------
//Fitness=817.175000
//Makespan=108.160
//Cost=1526.190
//Utilization=0.941
//Throughput=0.462
//AvgResponse=10.175
//AvgWaiting=50.741
//LoadBalance=6.152
//=== LocustDriver finished ===
