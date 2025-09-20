package org.workflowsim.examples.scheduling;

import java.io.File;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.text.DecimalFormat;
import java.util.*;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.*;
import org.workflowsim.Task;
import org.workflowsim.utils.Parameters;
import org.workflowsim.utils.ReplicaCatalog;

/**
 * WorkflowSimEvaluator with a cheap surrogate used for internal optimization
 * and an option to run a full CloudSim simulation for final verification.
 *
 * Usage:
 *  - During optimization set WorkflowSimEvaluator.FULL_SIM = false  (fast surrogate)
 *  - For final verification: WorkflowSimEvaluator.FULL_SIM = true (runs CloudSim once, prints results)
 */
public class WorkflowSimEvaluator {

    // DAX path taken from driver
    static String daxPath = LocustDriver.DAX_PATH;

    // Surrogate and VM sizing defaults (tweak if needed)
    private static final int VM_MIPS = 1000;   // used for surrogate and VM creation
    private static final int VM_PES = 1;
    private static final int VM_RAM = 2048;    // per-VM RAM
    private static final long VM_BW = 10000;
    private static final long VM_SIZE = 100000;
    private static final String VM_VMM = "Xen";

    private static final double COST_PER_SECOND = 3.0;
    private static final double wM = 0.5;
    private static final double wC = 0.5;

    /**
     * If FULL_SIM==false: evaluator uses surrogate (fast, no CloudSim).
     * If FULL_SIM==true: evaluator runs a real CloudSim simulation (slower), prints cloudlets.
     */
    public static boolean FULL_SIM = false;

    /**
     * Control whether the full-sim prints cloudlet details. Full simulation will print if this is true.
     * Note: surrogate never prints CloudSim details.
     */
    public static boolean PRINT_CLOUDLETS = false;

    /** Public entry: evaluate a Schedule's assignment[] and return metrics. */
    public static Result evaluateAssignment(int[] assignment) {
        try {
            if (!FULL_SIM) {
                // Surrogate estimator: very fast
                return surrogateEvaluation(assignment);
            } else {
                // Full CloudSim run (slower, accurate)
                return runWorkflowSimWithDax(assignment);
            }
        } catch (Throwable t) {
            t.printStackTrace();
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }
    }

    /**
     * Cheap surrogate: sum MI per VM, execTime = sumMI / VM_MIPS.
     * Makespan = max(execTime across VMs). Cost = sum(execTime)*COST_PER_SECOND.
     * This is *not* aware of precedence or data transfers but is very cheap.
     */
    private static Result surrogateEvaluation(int[] assignment) throws Exception {
        // Parse DAX to get task lengths (no CloudSim init required)
        File f = new File(daxPath);
        if (!f.exists()) {
            System.err.println("DAX not found for surrogate: " + daxPath);
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }
        Object parser = createAndParseWorkflowParser(daxPath, 0);
        List<Task> tasks = extractTaskListFromParser(parser);
        if (tasks == null || tasks.isEmpty()) {
            System.err.println("Surrogate: no tasks parsed from DAX.");
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }

        int vmCount = Math.max(Arrays.stream(assignment).max().orElse(0) + 1, 1);
        double[] sumMiPerVm = new double[vmCount];
        Arrays.fill(sumMiPerVm, 0.0);

        for (int i = 0; i < tasks.size(); i++) {
            Task t = tasks.get(i);
            long length = t.getCloudletLength(); // assuming Task inherits Cloudlet methods
            int vmIdx = i < assignment.length ? assignment[i] : assignment[i % assignment.length];
            vmIdx = Math.max(0, Math.min(vmIdx, vmCount - 1));
            sumMiPerVm[vmIdx] += length;
        }

        double makespan = 0.0;
        double totalCost = 0.0;
        for (int v = 0; v < vmCount; v++) {
            double execSec = sumMiPerVm[v] / (double) VM_MIPS;
            makespan = Math.max(makespan, execSec);
            totalCost += execSec * COST_PER_SECOND;
        }

        double mNorm = makespan / (makespan + 1e-9);
        double cNorm = totalCost / (totalCost + 1e-9);
        double fitness = wM * mNorm + wC * cNorm;
        return new Result(makespan, totalCost, fitness);
    }

    /** Run WorkflowSim (real simulation). Careful: this is slower. */
    private static Result runWorkflowSimWithDax(int[] assignment) throws Exception {
        File daxFile = new File(daxPath);
        if (!daxFile.exists()) {
            System.err.println("DAX file not found: " + daxPath);
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }

        // Initialize CloudSim fresh for this run
        CloudSim.init(1, Calendar.getInstance(), false);

        // Ensure ReplicaCatalog is ready for WorkflowParser
        ReplicaCatalog.init(ReplicaCatalog.FileSystem.LOCAL);

        // Parse workflow using WorkflowParser (it reads Parameters.getDaxPath())
        Object parser = createAndParseWorkflowParser(daxFile.getAbsolutePath(), 0);
        List<Task> taskList = extractTaskListFromParser(parser);
        if (taskList == null || taskList.isEmpty()) {
            System.err.println("FullSim: No tasks parsed from DAX.");
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }

        // Determine VM count from assignment
        int vmCount = Math.max(Arrays.stream(assignment).max().orElse(0) + 1, 1);

        // Create a datacenter sized enough to host all VM_PEs and per-VM RAM
        // Make host RAM >= vmCount * VM_RAM + overhead
        int hostRam = Math.max(16384, vmCount * VM_RAM + 1024);
        Datacenter datacenter = createDatacenterSized("Datacenter_0", vmCount, VM_MIPS, hostRam);

        // Create broker
        DatacenterBroker broker = new DatacenterBroker("Broker");
        int brokerId = broker.getId();

        // Create VMs
        List<Vm> vmList = new ArrayList<>();
        for (int i = 0; i < vmCount; i++) {
            Vm vm = new Vm(i, brokerId, VM_MIPS, VM_PES, VM_RAM, VM_BW, VM_SIZE, VM_VMM,
                    new CloudletSchedulerTimeShared());
            vmList.add(vm);
        }
        broker.submitVmList(vmList);

        // Map tasks -> VMs
        List<Cloudlet> cloudletsToSubmit = new ArrayList<>();
        for (int i = 0; i < taskList.size(); i++) {
            Task t = taskList.get(i);
            int vmIdx = i < assignment.length ? assignment[i] : assignment[i % assignment.length];
            vmIdx = Math.max(0, Math.min(vmIdx, vmList.size() - 1));
            t.setVmId(vmList.get(vmIdx).getId());
            t.setUserId(brokerId);
            cloudletsToSubmit.add(t);
        }
        broker.submitCloudletList(cloudletsToSubmit);

        // Run simulation
        CloudSim.startSimulation();
        CloudSim.stopSimulation();

        // Retrieve finished
        List<Cloudlet> finished;
        try {
            Method m = broker.getClass().getMethod("getCloudletReceivedList");
            finished = (List<Cloudlet>) m.invoke(broker);
        } catch (NoSuchMethodException nsme) {
            try {
                Method m2 = broker.getClass().getMethod("getCloudletFinishedList");
                finished = (List<Cloudlet>) m2.invoke(broker);
            } catch (NoSuchMethodException e) {
                finished = Collections.emptyList();
            }
        }

        // Optionally print cloudlet details (only for final run)
        if (PRINT_CLOUDLETS) {
            printCloudletList(finished);
        }

        // Compute metrics
        double makespan = 0.0;
        double totalCost = 0.0;
        for (Cloudlet cl : finished) {
            makespan = Math.max(makespan, cl.getFinishTime());
            totalCost += COST_PER_SECOND * cl.getActualCPUTime();
        }

        double mNorm = makespan / (makespan + 1e-9);
        double cNorm = totalCost / (totalCost + 1e-9);
        double fitness = wM * mNorm + wC * cNorm;

        return new Result(makespan, totalCost, fitness);
    }

    /** Print finished cloudlets (nice table). */
    private static void printCloudletList(List<Cloudlet> list) {
        DecimalFormat dft = new DecimalFormat("###.##");
        String indent = "    ";
        System.out.println();
        System.out.println("========== OUTPUT ==========");
        System.out.println("Cloudlet ID" + indent + "STATUS" + indent + "DataCenterID" + indent +
                "VM ID" + indent + "CPU Time" + indent + "Start Time" + indent + "Finish Time");
        for (Cloudlet cloudlet : list) {
            String status = cloudlet.getCloudletStatus() == Cloudlet.SUCCESS ? "SUCCESS" : "FAILED";
            System.out.println(indent +
                    cloudlet.getCloudletId() + indent + indent +
                    status + indent + indent +
                    cloudlet.getResourceId() + indent + indent + indent +
                    cloudlet.getVmId() + indent + indent +
                    dft.format(cloudlet.getActualCPUTime()) + indent + indent +
                    dft.format(cloudlet.getExecStartTime()) + indent + indent + indent +
                    dft.format(cloudlet.getFinishTime()));
        }
    }

    /** Parse & run WorkflowParser via reflection. */
    private static Object createAndParseWorkflowParser(String daxAbsolutePath, int brokerId) throws Exception {
        Parameters.setDaxPath(daxAbsolutePath);
        Class<?> parserClass = Class.forName("org.workflowsim.WorkflowParser");
        Constructor<?> ctor = parserClass.getConstructor(int.class);
        Object parserInstance = ctor.newInstance(brokerId);
        Method parseMethod = parserClass.getMethod("parse");
        parseMethod.invoke(parserInstance);
        return parserInstance;
    }

    /** Extract Task list from parser instance */
    @SuppressWarnings("unchecked")
    private static List<Task> extractTaskListFromParser(Object parserInstance) throws Exception {
        Method getTaskListMethod = parserInstance.getClass().getMethod("getTaskList");
        List<Task> list = (List<Task>) getTaskListMethod.invoke(parserInstance);
        return list == null ? Collections.emptyList() : list;
    }

    /**
     * Create a datacenter sized to host vmCount VMs: host has vmCount PEs each with 'peMips' MIPS.
     * Also the host RAM is passed in.
     */
    private static Datacenter createDatacenterSized(String name, int vmCount, int peMips, int hostRam) throws Exception {
        List<Host> hostList = new ArrayList<>();
        List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < vmCount; i++) {
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
                "x86", "Linux", "Xen", hostList, 10.0, COST_PER_SECOND, 0.05, 0.001, 0.0
        );
        return new Datacenter(name, characteristics, new VmAllocationPolicySimple(hostList), new LinkedList<>(), 0);
    }

    /** Result holder. */
    public static class Result {
        public double makespan;
        public double cost;
        public double fitness;
        public Result(double m, double c, double f) { this.makespan = m; this.cost = c; this.fitness = f; }
        public String toString() { return String.format("makespan=%.3f cost=%.3f fitness=%.5f", makespan, cost, fitness); }
    }
}
