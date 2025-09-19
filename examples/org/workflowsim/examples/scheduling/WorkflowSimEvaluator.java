package org.workflowsim.examples.scheduling;

import java.io.File;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.*;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.provisioners.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.power.PowerHost;
import org.workflowsim.Task;
import org.workflowsim.utils.Parameters;
import org.workflowsim.utils.ReplicaCatalog;

/**
 * WorkflowSimEvaluator integrates with WorkflowSim to evaluate a given task-to-VM assignment.
 */
public class WorkflowSimEvaluator {

    private static final String daxPath = "/Users/User/Desktop/WorkflowSim-1.0-master/WorkflowSim-1.0-master/config/dax/Montage_100.xml";

    private static volatile double Mmin = 0.0, Mmax = 1.0;
    private static volatile double Cmin = 0.0, Cmax = 1.0;
    private static volatile boolean baselinesInitialized = false;

    private static final double wM = 0.5;
    private static final double wC = 0.5;
    private static final double COST_PER_SECOND = 3.0;

    /** Public entry: evaluate a Schedule's assignment[] and return metrics. */
    public static Result evaluateAssignment(int[] assignment) {
        try {
            // Initialize baselines once
            if (!baselinesInitialized) {
                Mmin = 0.0; Mmax = 1.0;
                Cmin = 0.0; Cmax = 1.0;
                baselinesInitialized = true;
            }

            Result r = runWorkflowSimWithDax(assignment);

            // Normalize metrics for fitness
            double mNorm = (r.makespan - Mmin) / (Mmax - Mmin + 1e-9);
            double cNorm = (r.cost - Cmin) / (Cmax - Cmin + 1e-9);
            r.fitness = wM * mNorm + wC * cNorm;

            return r;

        } catch (Throwable t) {
            t.printStackTrace();
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }
    }

    /** Run WorkflowSim with a given DAX and assignment array */
    private static Result runWorkflowSimWithDax(int[] assignment) throws Exception {
        File daxFile = new File(daxPath);
        if (!daxFile.exists()) {
            System.err.println("DAX file not found: " + daxPath);
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }

        // Initialize CloudSim
        CloudSim.init(1, Calendar.getInstance(), false);

        // Initialize ReplicaCatalog properly
        ReplicaCatalog.init(ReplicaCatalog.FileSystem.SHARED);

        Datacenter datacenter = createDatacenter("Datacenter_0", 1);
        DatacenterBroker broker = createBroker();
        int brokerId = broker.getId();

        // Parse workflow
        Object parser = createAndParseWorkflowParser(daxFile.getAbsolutePath(), brokerId);

        // Extract tasks and convert to Cloudlet list
        List<Task> taskList = extractTaskListFromParser(parser);
        List<Cloudlet> jobList = new ArrayList<>();
        for (Task t : taskList) {
            t.setUserId(brokerId);
            jobList.add(t);
        }

        if (jobList.isEmpty()) {
            System.err.println("No jobs parsed from DAX.");
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }

        // Determine VM count
        int vmCount = Math.max(Arrays.stream(assignment).max().orElse(0) + 1, 1);
        List<Vm> vmList = new ArrayList<>();
        for (int i = 0; i < vmCount; i++) {
            Vm vm = new Vm(i, brokerId, 1000.0, 1, 2048, 10000, 100000, "Xen", new CloudletSchedulerTimeShared());
            vmList.add(vm);
        }
        broker.submitVmList(vmList);

        // Map tasks to VMs
        List<Cloudlet> cloudletsToSubmit = new ArrayList<>();
        for (int i = 0; i < jobList.size(); i++) {
            Cloudlet cl = jobList.get(i);
            int vmIdx = i < assignment.length ? assignment[i] : assignment[i % assignment.length];
            vmIdx = Math.max(0, Math.min(vmIdx, vmCount - 1));
            cl.setVmId(vmList.get(vmIdx).getId());
            cl.setUserId(brokerId);
            cloudletsToSubmit.add(cl);
        }
        broker.submitCloudletList(cloudletsToSubmit);

        // Run simulation
        CloudSim.startSimulation();
        CloudSim.stopSimulation();

        // Get finished cloudlets
        List<Cloudlet> finished;
        try {
            Method m = broker.getClass().getMethod("getCloudletReceivedList");
            finished = (List<Cloudlet>) m.invoke(broker);
        } catch (NoSuchMethodException nsme) {
            finished = Collections.emptyList();
        }

        double makespan = 0.0;
        double totalCost = 0.0;
        for (Cloudlet cl : finished) {
            makespan = Math.max(makespan, cl.getFinishTime());
            totalCost += COST_PER_SECOND * cl.getActualCPUTime();
        }

        return new Result(makespan, totalCost, 0.0);
    }

    /** Instantiate WorkflowParser and parse the DAX. */
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
        return (List<Task>) getTaskListMethod.invoke(parserInstance);
    }

    /** Create a classic datacenter. */
    private static Datacenter createDatacenter(String name, int numHosts) throws Exception {
        List<Host> hostList = new ArrayList<>();
        for (int i = 0; i < numHosts; i++) {
            List<Pe> peList = new ArrayList<>();
            peList.add(new Pe(0, new PeProvisionerSimple(1000)));
            Host host = new Host(
                i,
                new RamProvisionerSimple(16384),
                new BwProvisionerSimple(100000),
                1000000,
                peList,
                new VmSchedulerTimeShared(peList)
            );
            hostList.add(host);
        }
        DatacenterCharacteristics characteristics = new DatacenterCharacteristics(
            "x86", "Linux", "Xen", hostList, 10.0, COST_PER_SECOND, 0.05, 0.001, 0.0
        );
        return new Datacenter(name, characteristics, new VmAllocationPolicySimple(hostList), new LinkedList<>(), 0);
    }

    /** Create a broker. */
    private static DatacenterBroker createBroker() throws Exception {
        return new DatacenterBroker("Broker");
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
