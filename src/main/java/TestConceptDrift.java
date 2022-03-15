import com.opencsv.CSVWriter;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Prediction;
import moa.classifiers.core.driftdetection.*;
import moa.classifiers.rules.core.changedetection.NoChangeDetection;
import moa.core.Example;
import moa.core.TimingUtils;
import moa.evaluation.BasicConceptDriftPerformanceEvaluator;
import moa.learners.ChangeDetectorLearner;
import moa.options.ClassOption;
import moa.streams.generators.cd.AbruptChangeGenerator;
import moa.streams.generators.cd.AbstractConceptDriftGenerator;
import moa.streams.generators.cd.GradualChangeGenerator;

import java.io.FileWriter;
import java.io.IOException;

public class TestConceptDrift {

    public static AbstractConceptDriftGenerator stream = new AbruptChangeGenerator();

    public static <T extends ChangeDetector> void run(T method) {
        try {
            CSVWriter writer = new CSVWriter(new FileWriter("./data/cd/output_" + method.getClass().getSimpleName() + ".csv"));
            CSVWriter metrics = new CSVWriter(new FileWriter("./data/cd/metrics.csv", true));

            ChangeDetectorLearner learner = new ChangeDetectorLearner();
            BasicConceptDriftPerformanceEvaluator evaluator = new BasicConceptDriftPerformanceEvaluator();

            stream.restart();
            stream.prepareForUse();

            learner.driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd', "Drift detection method to use.", method.getClass(), method.getClass().getSimpleName());
            learner.setModelContext(stream.getHeader());
            learner.prepareForUse();

            int numberSamples = 0;
            int numberSamplesCorrect = 0;
            long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
            Example<Instance> trainInst;

            while(stream.hasMoreInstances() && numberSamples < 100000) {
                trainInst = stream.nextInstance();
                numberSamples++;

                learner.trainOnInstance(trainInst);

                if (learner.correctlyClassifies(trainInst.getData())) {
                    numberSamplesCorrect++;
                }

                Prediction pred = learner.getPredictionForInstance(trainInst);

                writer.writeNext(new String[]{String.valueOf(pred.getVotes()[0])}, false);
                evaluator.addResult(trainInst, pred);
            }

            writer.close();

            double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
            System.out.println(numberSamples + " instances in "+time+" seconds with " + method.getClass().getSimpleName() + ".");
            metrics.writeNext(new String[]{method.getClass().getSimpleName(), String.valueOf(time), String.valueOf(evaluator.getPredictionError()), String.valueOf(evaluator.getNumberWarnings()), String.valueOf(evaluator.getNumberDetections()), String.valueOf(evaluator.getNumberChanges()), String.valueOf(evaluator.getNumberChangesOccurred())}, false);
            metrics.close();
            System.out.println(evaluator.getNumberDetections());
            System.out.println(numberSamplesCorrect);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        run(new DDM());
        run(new EDDM());
        run(new EWMAChartDM());
        run(new EnsembleDriftDetectionMethods());
        run(new GeometricMovingAverageDM());
        run(new HDDM_A_Test());
        run(new HDDM_W_Test());
        run(new PageHinkleyDM());
        run(new RDDM());
        run(new SEEDChangeDetector());
        run(new STEPD());
        run(new SeqDrift1ChangeDetector());
        run(new SeqDrift2ChangeDetector());
        run(new NoChangeDetection());
    }

}
