import { Module } from '@nestjs/common';
import { BundlesController } from './bundles.controller';
import { BundlesService } from './bundles.service';

@Module({
  controllers: [BundlesController],
  providers: [BundlesService],
})
export class BundlesModule {}
