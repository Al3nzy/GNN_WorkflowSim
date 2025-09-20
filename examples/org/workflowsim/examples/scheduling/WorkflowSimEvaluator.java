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

public class WorkflowSimEvaluator {

    static String daxPath = LocustDriver.DAX_PATH;

    private static final int VM_MIPS = 1000;
    private static final int VM_PES = 1;
    private static final int VM_RAM = 512;
    private static final long VM_BW = 1000;
    private static final long VM_SIZE = 100000;
    private static final String VM_VMM = "Xen";

    private static final double COST_PER_SECOND = 3.0;

    private static final double wM = 0.5;
    private static final double wC = 0.5;

    public static boolean FULL_SIM = false;
    public static boolean PRINT_CLOUDLETS = false;

    public static Result evaluateAssignment(int[] assignment) {
        try {
            if (!FULL_SIM) {
                return surrogateEvaluation(assignment);
            } else {
                return runWorkflowSimWithDax(assignment);
            }
        } catch (Throwable t) {
            t.printStackTrace();
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }
    }

    // ---------------- Surrogate ----------------
    private static Result surrogateEvaluation(int[] assignment) throws Exception {
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

        for (int i = 0; i < tasks.size(); i++) {
            Task t = tasks.get(i);
            long length = t.getCloudletLength();
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

        double fitness = wM * makespan + wC * totalCost;
        return new Result(makespan, totalCost, fitness);
    }

    // ---------------- Full CloudSim ----------------
    private static Result runWorkflowSimWithDax(int[] assignment) throws Exception {
        File daxFile = new File(daxPath);
        if (!daxFile.exists()) {
            System.err.println("DAX file not found: " + daxPath);
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }

        CloudSim.init(1, Calendar.getInstance(), false);
        ReplicaCatalog.init(ReplicaCatalog.FileSystem.LOCAL);

        Object parser = createAndParseWorkflowParser(daxFile.getAbsolutePath(), 0);
        List<Task> taskList = extractTaskListFromParser(parser);
        if (taskList == null || taskList.isEmpty()) {
            System.err.println("FullSim: No tasks parsed from DAX.");
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }

        int vmCount = Math.max(Arrays.stream(assignment).max().orElse(0) + 1, 1);

        int hostRam = Math.max(16384, vmCount * VM_RAM + 1024);
        Datacenter datacenter = createDatacenterSized("Datacenter_0", vmCount, VM_MIPS, hostRam);

        DatacenterBroker broker = new DatacenterBroker("Broker");
        int brokerId = broker.getId();

        List<Vm> vmList = new ArrayList<>();
        for (int i = 0; i < vmCount; i++) {
            Vm vm = new Vm(i, brokerId, VM_MIPS, VM_PES, VM_RAM, VM_BW, VM_SIZE, VM_VMM,
                    new CloudletSchedulerSpaceShared());
            vmList.add(vm);
        }
        broker.submitVmList(vmList);

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

        CloudSim.startSimulation();
        CloudSim.stopSimulation();

        List<Cloudlet> finished;
        try {
            Method m = broker.getClass().getMethod("getCloudletReceivedList");
            finished = (List<Cloudlet>) m.invoke(broker);
        } catch (NoSuchMethodException nsme) {
            Method m2 = broker.getClass().getMethod("getCloudletFinishedList");
            finished = (List<Cloudlet>) m2.invoke(broker);
        }

        if (PRINT_CLOUDLETS) {
            printCloudletList(finished);
        }

        // ---- Compute extended metrics ----
        double makespan = 0;
        double totalCpu = 0;
        double totalResponseTime = 0;
        double totalWaitingTime = 0;
        int jobCount = finished.size();
        double[] vmLoad = new double[vmCount];

        for (Cloudlet cl : finished) {
            makespan = Math.max(makespan, cl.getFinishTime());
            totalCpu += cl.getActualCPUTime();
            double response = cl.getFinishTime() - cl.getExecStartTime();
            double waiting = cl.getExecStartTime(); // since submission ~0
            totalResponseTime += response;
            totalWaitingTime += waiting;

            if (cl.getVmId() >= 0 && cl.getVmId() < vmCount) {
                vmLoad[cl.getVmId()] += cl.getActualCPUTime();
            }
        }

        double cost = totalCpu * COST_PER_SECOND;
        double utilization = totalCpu / (makespan * vmCount);
        double throughput = jobCount / makespan;
        double avgResponse = totalResponseTime / jobCount;
        double avgWaiting = totalWaitingTime / jobCount;

        double avgLoad = totalCpu / vmCount;
        double sumSq = 0;
        for (double load : vmLoad) {
            sumSq += Math.pow(load - avgLoad, 2);
        }
        double loadBalance = Math.sqrt(sumSq / vmCount);

        double fitness = wM * makespan + wC * cost;

        return new Result(makespan, cost, fitness,
                utilization, throughput, avgResponse, avgWaiting, loadBalance);
    }

    // ---------------- Helpers ----------------
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

    private static Object createAndParseWorkflowParser(String daxAbsolutePath, int brokerId) throws Exception {
        Parameters.setDaxPath(daxAbsolutePath);
        Class<?> parserClass = Class.forName("org.workflowsim.WorkflowParser");
        Constructor<?> ctor = parserClass.getConstructor(int.class);
        Object parserInstance = ctor.newInstance(brokerId);
        Method parseMethod = parserClass.getMethod("parse");
        parseMethod.invoke(parserInstance);
        return parserInstance;
    }

    @SuppressWarnings("unchecked")
    private static List<Task> extractTaskListFromParser(Object parserInstance) throws Exception {
        Method getTaskListMethod = parserInstance.getClass().getMethod("getTaskList");
        return (List<Task>) getTaskListMethod.invoke(parserInstance);
    }

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
                "x86", "Linux", "Xen", hostList,
                10.0, COST_PER_SECOND, 0.05, 0.001, 0.0
        );
        return new Datacenter(name, characteristics, new VmAllocationPolicySimple(hostList), new LinkedList<>(), 0);
    }

    // ---------------- Result ----------------
    public static class Result {
        public double makespan, cost, fitness;
        public double utilization, throughput, avgResponse, avgWaiting, loadBalance;

        public Result(double m, double c, double f) {
            this(m, c, f, 0, 0, 0, 0, 0);
        }

        public Result(double m, double c, double f,
                      double util, double thr, double resp, double wait, double lb) {
            this.makespan = m;
            this.cost = c;
            this.fitness = f;
            this.utilization = util;
            this.throughput = thr;
            this.avgResponse = resp;
            this.avgWaiting = wait;
            this.loadBalance = lb;
        }

        public String toString() {
            return String.format(
                "makespan=%.3f cost=%.3f fitness=%.3f util=%.3f thr=%.3f resp=%.3f wait=%.3f lb=%.3f",
                makespan, cost, fitness, utilization, throughput, avgResponse, avgWaiting, loadBalance
            );
        }
    }
}
