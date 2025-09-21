package org.workflowsim.examples.scheduling;

import java.io.File;
import java.text.DecimalFormat;
import java.util.*;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.DatacenterBroker;
import org.cloudbus.cloudsim.DatacenterCharacteristics;
import org.cloudbus.cloudsim.Datacenter;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.*;
import org.cloudbus.cloudsim.VmSchedulerTimeShared;
import org.cloudbus.cloudsim.CloudletSchedulerSpaceShared;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.VmAllocationPolicySimple;

import org.workflowsim.WorkflowParser;
import org.workflowsim.Task;
import org.workflowsim.utils.Parameters;
import org.workflowsim.utils.ReplicaCatalog;

/**
 * WorkflowSimEvaluator: surrogate + single-run full CloudSim evaluation.
 *
 * Fixed printing to use getCloudletId() (compatible with CloudSim versions
 * where Cloudlet.getId() does not exist).
 */
public class WorkflowSimEvaluator {

    // Set by the driver
    public static String daxPath = LocustDriver.DAX_PATH;

    // VM defaults
    private static final int VM_MIPS = 1000;
    private static final int VM_PES = 1;
    private static final int VM_RAM = 512;
    private static final long VM_BW = 1000L;
    private static final long VM_SIZE = 100000L;
    private static final String VM_VMM = "Xen";

    private static final double COST_PER_SECOND = 3.0;

    // weights for normalized fitness
    private static final double wM = 0.5;
    private static final double wC = 0.5;

    // toggles
    public static boolean FULL_SIM = false;         // false => surrogate evaluation (fast)
    public static boolean PRINT_JOB_TABLE = true;   // prints workflow-style job table in full sim

    /**
     * Public entry. assignment: mapping task index -> vm index.
     * vmCount: number of VMs to create when running full simulation.
     */
    public static Result evaluateAssignment(int[] assignment, int vmCount) {
        try {
            if (!FULL_SIM) {
                return surrogateEvaluation(assignment, vmCount);
            } else {
                return fullSimulationEvaluate(assignment, vmCount);
            }
        } catch (Throwable t) {
            t.printStackTrace();
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }
    }

    // ---------------- Surrogate (fast) ----------------
    private static Result surrogateEvaluation(int[] assignment, int vmCount) throws Exception {
        File f = new File(daxPath);
        if (!f.exists()) {
            System.err.println("Surrogate: DAX not found: " + daxPath);
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }

        // parse DAX to get tasks (lengths)
        Parameters.setDaxPath(daxPath);
        WorkflowParser parser = new WorkflowParser(0);
        parser.parse();
        List<Task> tasks = parser.getTaskList();
        if (tasks == null || tasks.isEmpty()) {
            System.err.println("Surrogate: no tasks parsed.");
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }

        int vmNum = Math.max(1, vmCount);
        double[] sumMiPerVm = new double[vmNum];
        long totalMi = 0;

        for (int i = 0; i < tasks.size(); i++) {
            Task t = tasks.get(i);
            long len = t.getCloudletLength();
            totalMi += len;
            int vmIdx = (i < assignment.length) ? assignment[i] : assignment[i % assignment.length];
            vmIdx = Math.max(0, Math.min(vmIdx, vmNum - 1));
            sumMiPerVm[vmIdx] += len;
        }

        double makespan = 0.0;
        double totalCost = 0.0;
        for (int v = 0; v < vmNum; v++) {
            double secs = sumMiPerVm[v] / (double) VM_MIPS;
            makespan = Math.max(makespan, secs);
            totalCost += secs * COST_PER_SECOND;
        }

        // normalization bounds (algorithm-agnostic)
        double Mmin = totalMi / (double) (vmNum * VM_MIPS);
        double Mmax = totalMi / (double) VM_MIPS;
        double Cmin = Mmin * COST_PER_SECOND;
        double Cmax = Mmax * COST_PER_SECOND;

        double mNorm = safeNormalize(makespan, Mmin, Mmax);
        double cNorm = safeNormalize(totalCost, Cmin, Cmax);
        double fitness = wM * mNorm + wC * cNorm;

        return new Result(makespan, totalCost, fitness, mNorm, cNorm, 0, 0, 0, 0, 0.0);
    }

    // ---------------- Full single-run CloudSim evaluation (broker + cloudlets) ----------------
    @SuppressWarnings("unchecked")
    private static Result fullSimulationEvaluate(int[] assignment, int vmCount) throws Exception {
        File f = new File(daxPath);
        if (!f.exists()) {
            System.err.println("FullSim: DAX not found: " + daxPath);
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }

        // Initialize CloudSim for this single run
        CloudSim.init(1, Calendar.getInstance(), false);
        ReplicaCatalog.init(ReplicaCatalog.FileSystem.LOCAL);

        // Parse DAX
        Parameters.setDaxPath(daxPath);
        WorkflowParser parser = new WorkflowParser(0);
        parser.parse();
        List<Task> taskList = parser.getTaskList();
        if (taskList == null || taskList.isEmpty()) {
            System.err.println("FullSim: no tasks parsed.");
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }

        // Create datacenter sized to vmCount
        Datacenter datacenter = createDatacenterSized("Datacenter_0", Math.max(1, vmCount), VM_MIPS, Math.max(16384, vmCount * VM_RAM + 1024));

        // Create broker and VMs
        DatacenterBroker broker = new DatacenterBroker("Broker");
        int brokerId = broker.getId();

        List<Vm> vmList = new ArrayList<>();
        for (int i = 0; i < Math.max(1, vmCount); i++) {
            Vm vm = new Vm(i, brokerId, VM_MIPS, VM_PES, VM_RAM, VM_BW, VM_SIZE, VM_VMM,
                    new CloudletSchedulerSpaceShared());
            vmList.add(vm);
        }
        broker.submitVmList(vmList);

        // Map tasks to VMs and submit them
        List<Cloudlet> cloudletsToSubmit = new ArrayList<>();
        for (int i = 0; i < taskList.size(); i++) {
            Task t = taskList.get(i);
            int vmIdx = (i < assignment.length) ? assignment[i] : assignment[i % assignment.length];
            vmIdx = Math.max(0, Math.min(vmIdx, vmList.size() - 1));
            t.setVmId(vmList.get(vmIdx).getId());
            t.setUserId(brokerId);
            cloudletsToSubmit.add(t);
        }
        broker.submitCloudletList(cloudletsToSubmit);

        // Single start/stop for the run
        CloudSim.startSimulation();
        CloudSim.stopSimulation();

        // Get finished cloudlets; broker methods vary by version
        List<Cloudlet> finished;
        try {
            finished = (List<Cloudlet>) broker.getClass().getMethod("getCloudletReceivedList").invoke(broker);
        } catch (NoSuchMethodException nsme) {
            finished = (List<Cloudlet>) broker.getClass().getMethod("getCloudletFinishedList").invoke(broker);
        }

        // Print in WorkflowSim-like Job/Task table format
        if (PRINT_JOB_TABLE) {
            printJobListCompatibleFromCloudlets(finished);
        }

        // Compute metrics
        double makespan = 0;
        double totalCpu = 0;
        double totalResponse = 0;
        double totalWaiting = 0;
        int jobCount = finished.size();
        double[] vmLoad = new double[Math.max(1, vmList.size())];
        long totalMi = 0;

        for (Cloudlet cl : finished) {
            makespan = Math.max(makespan, cl.getFinishTime());
            totalCpu += cl.getActualCPUTime();
            double response = cl.getFinishTime() - cl.getExecStartTime();
            double waiting = cl.getExecStartTime();
            totalResponse += response;
            totalWaiting += waiting;
            int vid = (int) cl.getVmId();
            if (vid >= 0 && vid < vmLoad.length) vmLoad[vid] += cl.getActualCPUTime();
            try { totalMi += cl.getCloudletLength(); } catch (Throwable ignore) {}
        }

        double cost = totalCpu * COST_PER_SECOND;
        double utilization = (makespan > 0 && vmList.size() > 0) ? (totalCpu / (makespan * vmList.size())) : 0;
        double throughput = (makespan > 0) ? (jobCount / makespan) : 0;
        double avgResponse = (jobCount > 0) ? (totalResponse / jobCount) : 0;
        double avgWaiting = (jobCount > 0) ? (totalWaiting / jobCount) : 0;

        double avgLoad = totalCpu / Math.max(1, vmList.size());
        double sumSq = 0;
        for (double load : vmLoad) sumSq += Math.pow(load - avgLoad, 2);
        double loadBalance = Math.sqrt(sumSq / Math.max(1, vmList.size()));

        // if totalMi unknown, sum from parsed tasks
        if (totalMi == 0) {
            for (Task t : taskList) totalMi += t.getCloudletLength();
        }

        double sumMips = vmList.size() * VM_MIPS;
        double minMips = VM_MIPS;
        double Mmin = (sumMips > 0) ? totalMi / sumMips : 0;
        double Mmax = (minMips > 0) ? totalMi / minMips : 0;
        double Cmin = Mmin * COST_PER_SECOND;
        double Cmax = Mmax * COST_PER_SECOND;

        double mNorm = safeNormalize(makespan, Mmin, Mmax);
        double cNorm = safeNormalize(cost, Cmin, Cmax);
        double fitness = wM * mNorm + wC * cNorm;

        return new Result(makespan, cost, fitness, mNorm, cNorm, utilization, throughput, avgResponse, avgWaiting, loadBalance);
    }

    // ---------------- Printing: mimic WorkflowSim printJobList (job-style) ----------------
    private static void printJobListCompatibleFromCloudlets(List<Cloudlet> list) {
        String indent = "    ";
        Log.printLine();
        Log.printLine("========== OUTPUT ==========");
        Log.printLine("Job ID" + indent + "Task ID" + indent + "STATUS" + indent
                + "Data center ID" + indent + "VM ID" + indent + indent
                + "Time" + indent + "Start Time" + indent + "Finish Time" + indent + "Depth");
        DecimalFormat dft = new DecimalFormat("###.##");

        for (Cloudlet cl : list) {
            // Job/Cloudlet ID (use getCloudletId())
            long jobId = -1;
            try {
                jobId = cl.getCloudletId();
            } catch (Throwable ignore) { jobId = -1; }

            // STATUS
            String status = "UNKNOWN";
            try {
                int st = cl.getStatus();
                status = (st == Cloudlet.SUCCESS) ? "SUCCESS" : "FAILED";
            } catch (Throwable e) {
                try {
                    int st2 = cl.getCloudletStatus();
                    status = (st2 == Cloudlet.SUCCESS) ? "SUCCESS" : "FAILED";
                } catch (Throwable ignore) {}
            }

            // Data center id and VM id
            int resId = -1;
            int vmId = -1;
            try { resId = (int) cl.getResourceId(); } catch (Throwable ignore) {}
            try { vmId = (int) cl.getVmId(); } catch (Throwable ignore) {}

            // CPU time / start / finish
            double cpu = 0.0, start = 0.0, finish = 0.0;
            try { cpu = cl.getActualCPUTime(); } catch (Throwable ignore) {}
            try { start = cl.getExecStartTime(); } catch (Throwable ignore) {}
            try { finish = cl.getFinishTime(); } catch (Throwable ignore) {}

            // Depth: if this Cloudlet is actually a WorkflowSim Task, get depth
            int depth = -1;
            if (cl instanceof Task) {
                try { depth = ((Task) cl).getDepth(); } catch (Throwable ignore) {}
            }

            // Task ID list: if Task, print its cloudlet id; otherwise same as job id
            StringBuilder taskIds = new StringBuilder();
            if (cl instanceof Task) {
                try { taskIds.append(((Task) cl).getCloudletId()); } catch (Throwable ignore) { taskIds.append(jobId); }
            } else {
                taskIds.append(jobId);
            }

            // Print row
            Log.print(indent + jobId + indent + indent);
            Log.print(taskIds.toString());
            Log.print(indent);

            if ("SUCCESS".equals(status)) {
                Log.print("SUCCESS");
                Log.printLine(indent + indent + resId + indent + indent + indent + vmId
                        + indent + indent + indent + dft.format(cpu)
                        + indent + indent + dft.format(start) + indent + indent + indent
                        + dft.format(finish) + indent + indent + indent + (depth >= 0 ? depth : ""));
            } else {
                Log.print("FAILED");
                Log.printLine(indent + indent + resId + indent + indent + indent + vmId
                        + indent + indent + indent + dft.format(cpu)
                        + indent + indent + dft.format(start) + indent + indent + indent
                        + dft.format(finish) + indent + indent + indent + (depth >= 0 ? depth : ""));
            }
        }
    }

    // ---------------- Datacenter helper ----------------
    private static Datacenter createDatacenterSized(String name, int vmCount, int peMips, int hostRam) throws Exception {
        List<Host> hostList = new ArrayList<>();
        List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < Math.max(1, vmCount); i++) {
            peList.add(new Pe(i, new PeProvisionerSimple(peMips)));
        }
        Host host = new Host(
                0,
                new RamProvisionerSimple(hostRam),
                new BwProvisionerSimple(100000),
                1000000,
                peList,
                new VmSchedulerTimeShared(peList)
        );
        hostList.add(host);

        DatacenterCharacteristics characteristics = new DatacenterCharacteristics(
                "x86", "Linux", "Xen", hostList,
                10.0, COST_PER_SECOND, 0.05, 0.001, 0.0
        );

        return new Datacenter(name, characteristics, new VmAllocationPolicySimple(hostList), new ArrayList<>(), 0);
    }

    private static double safeNormalize(double val, double min, double max) {
        if (Double.isNaN(min) || Double.isInfinite(min) || Double.isNaN(max) || Double.isInfinite(max)) return 0.0;
        if (Math.abs(max - min) < 1e-9) return 0.0;
        return (val - min) / (max - min);
    }

    // ---------------- Result holder ----------------
    public static class Result {
        public double makespan, cost, fitness;
        public double mNorm, cNorm;
        public double utilization, throughput, avgResponse, avgWaiting, loadBalance;

        public Result(double m, double c, double f) {
            this(m, c, f, 0, 0, 0, 0, 0, 0, 0);
        }

        public Result(double m, double c, double f, double mN, double cN,
                      double util, double thr, double resp, double wait, double lb) {
            this.makespan = m;
            this.cost = c;
            this.fitness = f;
            this.mNorm = mN;
            this.cNorm = cN;
            this.utilization = util;
            this.throughput = thr;
            this.avgResponse = resp;
            this.avgWaiting = wait;
            this.loadBalance = lb;
        }

        
        
        
        public String toString() {
        	
        	 DecimalFormat df1 = new DecimalFormat("###.###");
             System.out.println("Makespan       = " + df1.format(makespan));
             System.out.println("Cost           = " + df1.format(cost));
             System.out.println("Utilization    = " + df1.format(utilization));
             System.out.println("Throughput     = " + df1.format(throughput));
             System.out.println("Avg Response   = " + df1.format(avgResponse));
             System.out.println("Avg Waiting    = " + df1.format(avgWaiting));
             System.out.println("Load Balanceσ  = " + df1.format(loadBalance));
         
        	
        	
            return String.format(Locale.US,
                    "makespan=%.3f cost=%.3f fitness=%.3f mNorm=%.3f cNorm=%.3f util=%.3f thr=%.3f resp=%.3f wait=%.3f lb=%.3f",
                    makespan, cost, fitness, mNorm, cNorm, utilization, throughput, avgResponse, avgWaiting, loadBalance);
       
        
        
        }
        
    }
}
