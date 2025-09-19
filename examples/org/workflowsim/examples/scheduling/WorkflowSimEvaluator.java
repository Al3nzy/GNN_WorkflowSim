package org.workflowsim.examples.scheduling;

import java.io.File;
import java.lang.reflect.*;
import java.util.*;
import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.power.PowerHost;
import org.cloudbus.cloudsim.provisioners.*;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.CloudletSchedulerTimeShared;
import org.cloudbus.cloudsim.VmSchedulerTimeShared;
import org.cloudbus.cloudsim.VmAllocationPolicySimple;
import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.Cloudlet;
//import org.cloudbus.cloudsim.cloudlets.CloudletSimple;

/**
 * WorkflowSim evaluator that runs a DAX workflow via WorkflowSim/CloudSim.
 * Uses reflection for WorkflowParser to be compatible with several WorkflowSim versions.
 *
 * IMPORTANT:
 * - Put CloudSim & WorkflowSim jars on your project's classpath.
 * - Update daxPath to your actual DAX file path.
 */
public class WorkflowSimEvaluator {

    // Path to your DAX (update if needed)
    private static final String daxPath = "/Users/User/Desktop/WorkflowSim-1.0-master/WorkflowSim-1.0-master/config/dax/Montage_100.xml";

    // Normalization bounds (should be set using real baselines in a real experiment)
    private static volatile double Mmin = 0.0, Mmax = 1.0;
    private static volatile double Cmin = 0.0, Cmax = 1.0;
    private static volatile boolean baselinesInitialized = false;

    // weights for fitness
    private static final double wM = 0.5;
    private static final double wC = 0.5;

    // simple cost model (cost per second). You can make this read from DatacenterCharacteristics if desired.
    private static final double COST_PER_SECOND = 3.0;

    /** Public entry: evaluate a Schedule's assignment[] and return metrics. */
    public static Result evaluateAssignment(int[] assignment) {
        try {
            // initialize baselines once (optional)
            if (!baselinesInitialized) {
                // In a full experiment you should run HEFT/Min-Min here to populate bounds.
                // For now we keep default bounds to avoid division-by-zero.
                Mmin = 0.0; Mmax = 1.0;
                Cmin = 0.0; Cmax = 1.0;
                baselinesInitialized = true;
            }

            Result r = runWorkflowSimWithDax(assignment);

            // normalize
            double mNorm = (r.makespan - Mmin) / (Mmax - Mmin + 1e-9);
            double cNorm = (r.cost - Cmin) / (Cmax - Cmin + 1e-9);
            r.fitness = wM * mNorm + wC * cNorm;
            return r;

        } catch (Throwable t) {
            t.printStackTrace();
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }
    }

    /** Core: run CloudSim + WorkflowParser for the given assignment[] mapping. */
    private static Result runWorkflowSimWithDax(int[] assignment) throws Exception {
        // 1) Check DAX file
        File daxFile = new File(daxPath);
        if (!daxFile.exists()) {
            System.err.println("DAX file not found: " + daxPath);
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }

        // 2) Init CloudSim
        int numUser = 1;
        Calendar calendar = Calendar.getInstance();
        boolean traceFlag = false;
        CloudSim.init(numUser, calendar, traceFlag);

        // 3) Create Datacenter and Broker (classic API)
        Datacenter datacenter = createDatacenter("Datacenter_0", 1); // 1 host for simplicity
        DatacenterBroker broker = createBroker();
        int brokerId = broker.getId();

        // 4) Parse DAX reflectively and extract cloudlet/job list
        Object parserOrList = createAndParseWorkflowParser(daxFile.getAbsolutePath(), brokerId);
        List<Cloudlet> jobList = extractJobListFromParser(parserOrList);
        if (jobList == null || jobList.isEmpty()) {
            System.err.println("No jobs parsed from DAX or parser method not found.");
            return new Result(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
        }

        // 5) Determine number of VMs to create (based on assignment or at least 1)
        int vmCount = Math.max(Arrays.stream(assignment).max().orElse(0) + 1, 1);

        // 6) Create VMs and submit to broker (classic Vm constructor)
        List<Vm> vmList = new ArrayList<>();
        for (int i = 0; i < vmCount; i++) {
            // constructor: Vm(id, userId, mips, pesNumber, ram, bw, size, vmm, cloudletScheduler)
            Vm vm = new Vm(i, brokerId, 1000.0, 1, 2048, 10000, 100000, "Xen", new CloudletSchedulerTimeShared());
            vmList.add(vm);
        }
        broker.submitVmList(vmList);

        // 7) Map parsed jobs to VMs using assignment[] and set userId
        List<Cloudlet> cloudletsToSubmit = new ArrayList<>();
        int numJobs = jobList.size();
        for (int i = 0; i < numJobs; i++) {
            Cloudlet cl = jobList.get(i);

            // If job is an object that is not a Cloudlet but has methods similar to Cloudlet,
            // reflection would be required; many WorkflowSim Job classes extend Cloudlet.
            // Here we assume jobList elements are Cloudlet-compatible.

            int vmIdx;
            if (i < assignment.length) vmIdx = assignment[i];
            else vmIdx = assignment[i % assignment.length];

            if (vmIdx < 0) vmIdx = 0;
            if (vmIdx >= vmCount) vmIdx = vmIdx % vmCount;

            // set vm id to CloudSim VM id
            cl.setVmId(vmList.get(vmIdx).getId());
            cl.setUserId(brokerId);
            cloudletsToSubmit.add(cl);
        }

        broker.submitCloudletList(cloudletsToSubmit);

        // 8) Run the simulation
        CloudSim.startSimulation();
        CloudSim.stopSimulation();

        // 9) Collect results from broker
        List<Cloudlet> finished = null;
        try {
            // try common method names to get finished cloudlets
            Method m = broker.getClass().getMethod("getCloudletReceivedList");
            finished = (List<Cloudlet>) m.invoke(broker);
        } catch (NoSuchMethodException nsme) {
            try {
                Method m2 = broker.getClass().getMethod("getCloudletFinishedList");
                finished = (List<Cloudlet>) m2.invoke(broker);
            } catch (NoSuchMethodException nsme2) {
                // fallback: try field or other method
                finished = Collections.emptyList();
            }
        }

        double makespan = 0.0;
        double totalCost = 0.0;
        for (Cloudlet cl : finished) {
            makespan = Math.max(makespan, cl.getFinishTime());
            // Use simple cost model: cost per second * actual CPU time
            totalCost += COST_PER_SECOND * cl.getActualCPUTime();
        }

        return new Result(makespan, totalCost, 0.0);
    }

    /** Reflectively instantiate a WorkflowParser and call parse(...) (handles multiple signatures). */
    private static Object createAndParseWorkflowParser(String daxAbsolutePath, int brokerId) throws Exception {
        String[] possibleNames = {
            "org.workflowsim.parser.WorkflowParser",
            "org.workflowsim.parser.DAXParser",
            "org.workflowsim.parser.Parser",
            "org.workflowsim.utils.PegasusParser"   // <--- add this
        };

        Class<?> parserClass = null;
        for (String cname : possibleNames) {
            try {
                parserClass = Class.forName(cname);
                break;
            } catch (ClassNotFoundException e) {
                // try next
            }
        }
        if (parserClass == null) {
            throw new RuntimeException("Parser class not found in WorkflowSim jar. Checked: " + Arrays.toString(possibleNames));
        }

        // PegasusParser is usually constructed with (String daxPath, int userId)
        Object parserInstance;
        try {
            Constructor<?> ctor = parserClass.getConstructor(String.class, int.class);
            parserInstance = ctor.newInstance(daxAbsolutePath, brokerId);
        } catch (NoSuchMethodException e) {
            // fallback to (String) constructor
            Constructor<?> ctor = parserClass.getConstructor(String.class);
            parserInstance = ctor.newInstance(daxAbsolutePath);
        }

        // call parse()
        Method parseMethod = parserClass.getMethod("parse");
        parseMethod.invoke(parserInstance);

        return parserInstance;
    }

    /** Extract a List<Cloudlet> from either a returned List or parser instance via common getters. */
    @SuppressWarnings("unchecked")
    private static List<Cloudlet> extractJobListFromParser(Object parserOrList) throws Exception {
        if (parserOrList == null) return Collections.emptyList();

        if (parserOrList instanceof List) {
            List<?> raw = (List<?>) parserOrList;
            List<Cloudlet> out = new ArrayList<>();
            for (Object o : raw) {
                if (o instanceof Cloudlet) out.add((Cloudlet) o);
                else {
                    // try cast via reflection
                    try {
                        out.add((Cloudlet) o);
                    } catch (ClassCastException cce) {
                        // skip
                    }
                }
            }
            return out;
        }

        Class<?> cls = parserOrList.getClass();
        String[] getterNames = {"getJobList", "getJobs", "getCloudletList", "getJobListById", "getCloudletListById", "getList"};
        for (String name : getterNames) {
            try {
                Method gm = cls.getMethod(name);
                Object ret = gm.invoke(parserOrList);
                if (ret instanceof List) {
                    List<?> raw = (List<?>) ret;
                    List<Cloudlet> out = new ArrayList<>();
                    for (Object o : raw) {
                        if (o instanceof Cloudlet) out.add((Cloudlet) o);
                        else {
                            try {
                                out.add((Cloudlet) o);
                            } catch (ClassCastException cce) { /* skip */ }
                        }
                    }
                    if (!out.isEmpty()) return out;
                }
            } catch (NoSuchMethodException nsme) {
                // ignore
            }
        }

        throw new RuntimeException("Could not extract job list from parser: no known getter found.");
    }

    /** Create a classic Datacenter (not DatacenterSimple) using Host and Pe objects. */
    private static Datacenter createDatacenter(String name, int numHosts) throws Exception {
        List<Host> hostList = new ArrayList<>();

        for (int i = 0; i < numHosts; i++) {
            List<Pe> peList = new ArrayList<>();
            peList.add(new Pe(0, new PeProvisionerSimple(1000)));

            int hostId = i;
            int ram = 16384; // MB
            long storage = 1000000; // MB
            int bw = 100000;

            Host host = new Host(
                    hostId,
                    new RamProvisionerSimple(ram),
                    new BwProvisionerSimple(bw),
                    storage,
                    peList,
                    new VmSchedulerTimeShared(peList)
            );
            hostList.add(host);
        }

        String arch = "x86";
        String os = "Linux";
        String vmm = "Xen";
        double timeZone = 10.0;
        double costPerSec = COST_PER_SECOND;
        double costPerMem = 0.05;
        double costPerStorage = 0.001;
        double costPerBw = 0.0;

        DatacenterCharacteristics characteristics =
                new DatacenterCharacteristics(arch, os, vmm, hostList, timeZone, costPerSec, costPerMem, costPerStorage, costPerBw);

        Datacenter datacenter = new Datacenter(
                name,
                characteristics,
                new VmAllocationPolicySimple(hostList),
                new LinkedList<>(),
                0
        );
        return datacenter;
    }

    /** Classic DatacenterBroker creation. */
    private static DatacenterBroker createBroker() throws Exception {
        return new DatacenterBroker("Broker");
    }

    /** Simple result holder. */
    public static class Result {
        public double makespan;
        public double cost;
        public double fitness;
        public Result(double m, double c, double f) {
            this.makespan = m;
            this.cost = c;
            this.fitness = f;
        }
        public String toString() {
            return String.format("makespan=%.3f cost=%.3f fitness=%.5f", makespan, cost, fitness);
        }
    }
}
