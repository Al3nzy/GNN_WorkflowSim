package org.workflowsim.examples.scheduling;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import org.workflowsim.Task;
import org.workflowsim.WorkflowParser;
import org.workflowsim.utils.ClusteringParameters;
import org.workflowsim.utils.OverheadParameters;
import org.workflowsim.utils.Parameters;
import org.workflowsim.utils.ReplicaCatalog;

/**
 * LocustDriver: prepare parameters, run Locust (surrogate), then final full simulation.
 */
public class LocustDriver {

    public static final String DAX_PATH = "examples/org/workflowsim/examples/workflowDatasets/Montage_1000.xml";

    public static void main(String[] args) {
        try {
            System.out.println("=== LocustDriver starting ===");

            File daxFile = new File(DAX_PATH);
            if (!daxFile.exists()) {
                System.err.println("DAX not found: " + DAX_PATH);
                return;
            }

            // Basic WorkflowSim params so WorkflowParser reads DAX properly
            int vmNum = 5;
            OverheadParameters op = new OverheadParameters(0, null, null, null, null, 0);
            ClusteringParameters cp = new ClusteringParameters(0, 0, ClusteringParameters.ClusteringMethod.NONE, null);

            Parameters.init(vmNum, DAX_PATH, null, null, op, cp, Parameters.SchedulingAlgorithm.INVALID, Parameters.PlanningAlgorithm.INVALID, null, 0L);
            ReplicaCatalog.init(ReplicaCatalog.FileSystem.LOCAL);

            // Count tasks from DAX
            int numTasks = parseTaskCount(0);
            System.out.println("Parsed number of tasks: " + numTasks);

            // Configure locust
            int pop = 30;
            int iter = 5; // increase for real runs
            LocustScheduler sched = new LocustScheduler(pop, iter, numTasks, vmNum, 1.0, 0.5, 0.3, 0.5, 0.2, 0.25, 0.1);

            // Use surrogate during search
            WorkflowSimEvaluator.FULL_SIM = false;
            System.out.println("Running Locust (surrogate) ...");
            sched.run();

            // Final: run a single full simulation with best assignment
            Schedule best = sched.getBestSchedule();
            if (best != null) {
                System.out.println("\n=== Final full simulation with best schedule ===");
                WorkflowSimEvaluator.FULL_SIM = true;
                WorkflowSimEvaluator.PRINT_JOB_TABLE = true;

                WorkflowSimEvaluator.Result finalRes = WorkflowSimEvaluator.evaluateAssignment(best.getAssignment(), vmNum);
                System.out.println("=== FINAL METRICS (Locust) ===");
                System.out.println(finalRes);
                System.out.println("Best assignment: " + Arrays.toString(best.getAssignment()));
            } else {
                System.err.println("No best schedule returned by Locust.");
            }

            System.out.println("=== LocustDriver finished ===");

        } catch (Throwable t) {
            t.printStackTrace();
        }
    }

    private static int parseTaskCount(int userId) throws Exception {
        Parameters.setDaxPath(DAX_PATH);
        WorkflowParser parser = new WorkflowParser(userId);
        parser.parse();
        List<Task> tasks = parser.getTaskList();
        return tasks == null ? 0 : tasks.size();
    }
}
